import os
from typing import TYPE_CHECKING, Union, cast, Optional, Tuple, List
import sys
import numpy as np
import open3d as o3d
import json
import habitat
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.core.agent import Agent
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)

# Set paths
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
submodules_dir = os.path.join(current_script_dir, 'habitat-lab')
sys.path.append(submodules_dir)

# Silent logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

# Set CPU rendering environment variables
# os.environ["EGL_DEVICE_ID"] = "-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# # Do not set MAGNUM_DEVICE to cpu, let it automatically choose the appropriate rendering backend
# # os.environ["MAGNUM_DEVICE"] = "cpu"
# os.environ["HABITAT_SIM_HEADLESS"] = "1"

if TYPE_CHECKING:
    from habitat.core.simulator import Observations
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


class ShortestPathFollowerAgent(Agent):
    """Shortest path following agent"""
    
    def __init__(self, env: habitat.Env, goal_radius: float):
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast("HabitatSim", env.sim),
            goal_radius=goal_radius,
            return_one_hot=False,
        )

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        return self.shortest_path_follower.get_next_action(
            cast(NavigationEpisode, self.env.current_episode).goals[0].position
        )

    def reset(self) -> None:
        pass


class NavigationAPI:
    """Habitat navigation API interface class

    Provides navigation functionality based on point clouds and semantic segmentation results, generating navigation videos.
    """
    
    def __init__(self, 
                 yaml_path: str = "config/benchmark/nav/pointnav/pointnav_scannet.yaml",
                 scene_path: Optional[str] = None,
                 start_position: Optional[List[float]] = None,
                 goal_radius: float = 0.2,
                 fps: int = 6,
                 video_quality: int = 9):
        """
        Initialize navigation API

        Args:
            yaml_path: Habitat configuration file path
            scene_path: Scene file path (.glb file)
            start_position: Starting position coordinates [x, y, z], if None, automatically calculated based on scene boundaries
            goal_radius: Target radius
            fps: Video frame rate
            video_quality: Video quality (1-10)
        """
        self.yaml_path = yaml_path
        self.scene_path = scene_path
        self.start_position = start_position  # No longer set default value, will be calculated at runtime
        self.goal_radius = goal_radius
        self.fps = fps
        self.video_quality = video_quality
        
    def _load_point_cloud(self, ply_path: str) -> np.ndarray:
        """Load and preprocess point cloud data"""
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        
        # Coordinate system translation: convert point cloud coordinate system from centered at 0 to starting from 0
        x_offset = -points[:, 0].min()
        y_offset = -points[:, 1].min()
        z_offset = 0
        
        points[:, 0] += x_offset
        points[:, 1] += y_offset
        points[:, 2] += z_offset
        
        return points
    
    def _load_segmentation_mask(self, jsonl_path: str) -> np.ndarray:
        """Load semantic segmentation mask"""
        with open(jsonl_path, "r") as f:
            for line in f:
                data = json.loads(line)
                return np.array(data["pred_mask"])
        raise ValueError("Unable to read segmentation mask from JSONL file")
    
    def _transform_coordinates(self, target_center: np.ndarray) -> np.ndarray:
        """Coordinate system transformation: point cloud coordinate system -> Habitat coordinate system"""
        # Point cloud: [x, y, z] -> Habitat: [x, z, -y]
        transformed = target_center.copy()
        transformed[1], transformed[2] = transformed[2], -transformed[1]
        return transformed
    
    def _create_habitat_config(self, scene_path: Optional[str] = None) -> habitat.config:
        """Create Habitat configuration"""
        config = habitat.get_config(config_path=self.yaml_path)
        
        with habitat.config.read_write(config):
            # If scene path is specified, override the scene settings in configuration
            if scene_path:
                # Convert to absolute path
                abs_scene_path = os.path.abspath(scene_path)
                config.habitat.simulator.scene = abs_scene_path
                
                # Update dataset configuration file
                self._update_dataset_config(abs_scene_path)
            
            # Configure CPU rendering mode, but keep sensors working normally
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1  # Use CPU
            config.habitat.simulator.habitat_sim_v0.enable_physics = True
                
            config.habitat.task.measurements.update({
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            })
        
        return config
    
    def _update_dataset_config(self, scene_path: str):
        """Update dataset configuration file, write scene_path into it"""
        import gzip
        
        dataset_path = "data/datasets/pointnav/v1/train/train1.json"
        dataset_gz_path = "data/datasets/pointnav/v1/train/train1.json.gz"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        
        # Read existing configuration or create new configuration
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r') as f:
                dataset_config = json.load(f)
        else:
            dataset_config = {"episodes": []}
        
        # Only update the scene_id field in the first dictionary of the episodes list
        if "episodes" in dataset_config and len(dataset_config["episodes"]) > 0:
            dataset_config["episodes"][0]["scene_id"] = scene_path
        
        # Write back to JSON file
        with open(dataset_path, 'w') as f:
            json.dump(dataset_config, f, indent=2)
        
        # Automatically generate compressed .gz file
        with open(dataset_path, 'rb') as f_in:
            with gzip.open(dataset_gz_path, 'wb') as f_out:
                f_out.write(f_in.read())
    
    def _create_custom_episode(self, scene_path: str, start_position: List[float], target_position: List[float]) -> NavigationEpisode:
        """Create custom navigation episode"""
        episode = NavigationEpisode(
            episode_id="0",
            scene_id=scene_path,
            start_position=start_position,
            start_rotation=[0, 0, 0, 1],  # Default orientation
            goals=[NavigationGoal(
                position=target_position,
                radius=self.goal_radius
            )]
        )
        return episode
    
    def _calculate_start_position(self, env: habitat.Env, fixed_height: float = 0.41561477) -> List[float]:
        """Calculate appropriate starting position based on scene boundaries"""
        lower, upper = env.sim.pathfinder.get_bounds()
        
        # Calculate center position for x and z axes, use fixed height for y axis
        # Ensure all values are Python native float types to avoid numpy type conversion errors
        center_x = float((lower[0] + upper[0]) / 2.0)
        center_z = float((lower[2] + upper[2]) / 2.0)
        start_pos = [center_x, float(fixed_height), center_z]
        
        # Check if the calculated position is navigable, try nearby positions if not navigable
        if not env.sim.pathfinder.is_navigable(start_pos):
            print(f"⚠️  Calculated center position is not navigable: {start_pos}, searching for nearby navigable positions...")
            
            # Search for navigable positions around the center position
            search_radius = 0.5
            search_step = 0.1
            found_navigable = False
            
            for offset_x in np.arange(-search_radius, search_radius + search_step, search_step):
                for offset_z in np.arange(-search_radius, search_radius + search_step, search_step):
                    test_pos = [float(center_x + offset_x), float(fixed_height), float(center_z + offset_z)]
                    # Ensure test position is within boundaries
                    if (lower[0] <= test_pos[0] <= upper[0] and 
                        lower[2] <= test_pos[2] <= upper[2] and
                        env.sim.pathfinder.is_navigable(test_pos)):
                        start_pos = test_pos
                        found_navigable = True
                        print(f"✅ Found navigable starting position: {start_pos}")
                        break
                if found_navigable:
                    break
            
            if not found_navigable:
                print(f"⚠️  No navigable starting position found, using default position within boundaries")
                # Use a relatively safe position within boundaries
                start_pos = [float(lower[0] + 1.0), float(fixed_height), float(lower[2] + 1.0)]
        else:
            print(f"✅ Calculated center position is navigable: {start_pos}")
        
        return start_pos
    
    def _validate_and_adjust_goal(self, target_center: np.ndarray, env: habitat.Env) -> np.ndarray:
        """Validate and adjust target coordinates within navigation mesh boundaries, and ensure position is navigable"""
        lower, upper = env.sim.pathfinder.get_bounds()
        print(f"Navigation mesh boundaries:")
        print(f"  Lower bound: {lower}")
        print(f"  Upper bound: {upper}")

        print(f"Target coordinates:",target_center)
        original_target = target_center.copy()
        
        # First adjust coordinates to within boundaries
        target_center[0] = np.clip(target_center[0], lower[0], upper[0])
        target_center[1] = np.clip(target_center[1], lower[1], upper[1])
        target_center[2] = np.clip(target_center[2], lower[2], upper[2])
        
        if not np.allclose(original_target, target_center):
            print(f"⚠️  Target coordinates adjusted to within boundaries:")
            print(f"  Original coordinates: {original_target}")
            print(f"  Adjusted coordinates: {target_center}")
        
        # Check if adjusted position is navigable
        target_pos = target_center.tolist()
        if env.sim.pathfinder.is_navigable(target_pos):
            print(f"✅ Target position is navigable: {target_pos}")
            return target_center
        
        print(f"⚠️  Target position is not navigable: {target_pos}, searching for nearest navigable position...")
        
        # Find nearest navigable position
        best_pos = target_center.copy()
        min_distance = float('inf')
        found_navigable = False
        
        # Search parameters
        max_search_radius = 2.0  # Maximum search radius
        search_step = 0.1  # Search step size
        
        for radius in np.arange(search_step, max_search_radius + search_step, search_step):
            # Search on sphere surface with current radius
            for theta in np.arange(0, 2 * np.pi, np.pi / 8):  # 8 directions
                for phi in np.arange(0, np.pi, np.pi / 4):  # 4 height levels
                    # Convert spherical coordinates to Cartesian coordinates
                    offset_x = radius * np.sin(phi) * np.cos(theta)
                    offset_y = radius * np.cos(phi)
                    offset_z = radius * np.sin(phi) * np.sin(theta)
                    
                    test_pos = [
                        float(target_center[0] + offset_x),
                        float(target_center[1] + offset_y),
                        float(target_center[2] + offset_z)
                    ]
                    
                    # Ensure test position is within boundaries
                    if (lower[0] <= test_pos[0] <= upper[0] and 
                        lower[1] <= test_pos[1] <= upper[1] and
                        lower[2] <= test_pos[2] <= upper[2]):
                        
                        if env.sim.pathfinder.is_navigable(test_pos):
                            # Calculate distance to original target
                            distance = np.linalg.norm(np.array(test_pos) - original_target)
                            if distance < min_distance:
                                min_distance = distance
                                best_pos = np.array(test_pos)
                                found_navigable = True
            
            # If navigable position found at current radius, do not expand search range
            if found_navigable:
                break
        
        if found_navigable:
            print(f"✅ Found nearest navigable position: {best_pos.tolist()}")
            print(f"   Distance to original target: {min_distance:.3f}")
            return best_pos
        else:
            print(f"⚠️  No navigable position found, using boundary-adjusted position: {target_center.tolist()}")
            return target_center
    
    def navigate_to_target_with_mask(self, 
                          ply_path: str, 
                          pred_mask: np.ndarray, 
                          output_path: str,
                          scene_path: Optional[str] = None,
                          video_name: Optional[str] = None) -> Tuple[str, dict]:
        """
                Execute navigation task and generate video (directly using mask)
        
        Args:
            ply_path: Point cloud file path (.ply)
            pred_mask: Semantic segmentation mask array
            output_path: Output directory path
            scene_path: Scene file path (.glb), if None, use scene_path from initialization or default scene in config file
            video_name: Custom video name, if None, automatically generated
            
        Returns:
            Tuple[str, dict]: (video file path, navigation statistics)
            
        Raises:
            ValueError: When no target points are detected or files do not exist
            FileNotFoundError: When input files do not exist
        """
        # Validate input files
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"Point cloud file does not exist: {ply_path}")
        
        # Load data
        points = self._load_point_cloud(ply_path)
        
        # Extract target points
        target_points = points[pred_mask == 1]
        if len(target_points) == 0:
            raise ValueError("No target points detected")
        
        target_center = target_points.mean(axis=0)
        target_center = self._transform_coordinates(target_center)
        
        # Determine the scene path to use
        used_scene_path = scene_path or self.scene_path
        if not used_scene_path:
            raise ValueError("Scene file path must be specified, either through scene_path parameter or scene_path parameter during initialization")
        
        # Validate if scene file exists
        if not os.path.exists(used_scene_path):
            raise FileNotFoundError(f"Scene file does not exist: {used_scene_path}")
        
        # Create configuration and environment
        config = self._create_habitat_config(used_scene_path)
        
        with habitat.Env(config=config) as env:
            # If no starting position specified, calculate based on scene boundaries
            if self.start_position is None:
                calculated_start_position = self._calculate_start_position(env)
            else:
                calculated_start_position = self.start_position.copy()
            
            # Navigation statistics
            nav_stats = {
                "target_points_count": len(target_points),
                "original_target_center": target_center.copy(),
                "start_position": calculated_start_position.copy(),
                "scene_path": used_scene_path
            }
            
            # Validate and adjust target coordinates
            target_center = self._validate_and_adjust_goal(target_center, env)
            nav_stats["adjusted_target_center"] = target_center.copy()
            
            # Create custom episode
            custom_episode = self._create_custom_episode(
                scene_path=used_scene_path,
                start_position=calculated_start_position,
                target_position=target_center.tolist()
            )
            
            # Manually set current episode
            env._current_episode = custom_episode
            
            # Target position in episode has been set during creation
            
            # Create agent
            agent = ShortestPathFollowerAgent(
                env=env,
                goal_radius=config.habitat.task.measurements.success.success_distance,
            )
            
            # Execute navigation
            observations = env.reset()
            agent.reset()
            
            vis_frames = []
            step_count = 0
            
            # Initial frame
            info = env.get_metrics()
            frame = observations_to_image(observations, info)
            info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            vis_frames.append(frame)
            
            # Navigation loop
            while not env.episode_over:
                action = agent.act(observations)
                if action is None:
                    break
                
                observations = env.step(action)
                info = env.get_metrics()
                frame = observations_to_image(observations, info)
                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)
                step_count += 1
            
            nav_stats["total_steps"] = step_count
            nav_stats["success"] = env.episode_over
            
            # Generate video
            if video_name is None:
                ply_filename = os.path.splitext(os.path.basename(ply_path))[0]
                scene_id = os.path.splitext(os.path.basename(custom_episode.scene_id))[0]
                video_name = f"{scene_id}_{custom_episode.episode_id}_{ply_filename}"
            
            os.makedirs(output_path, exist_ok=True)
            images_to_video(
                vis_frames, output_path, video_name, 
                fps=self.fps, quality=self.video_quality
            )
            
            # Manually construct video file path as images_to_video function does not return path
            video_name = video_name.replace(" ", "_").replace("\n", "_")
            video_name_split = video_name.split("/")
            video_name = "/".join(
                video_name_split[:-1] + [video_name_split[-1][:251] + ".mp4"]
            )
            video_path = os.path.join(output_path, video_name)
            
            nav_stats["video_path"] = video_path
            nav_stats["frames_count"] = len(vis_frames)
            
        return video_path, nav_stats

    def navigate_to_target(self, 
                          ply_path: str, 
                          jsonl_path: str, 
                          output_path: str,
                          scene_path: Optional[str] = None,
                          video_name: Optional[str] = None) -> Tuple[str, dict]:
        """
        Execute navigation task and generate video

        Args:
            ply_path: Point cloud file path (.ply)
            jsonl_path: Semantic segmentation result file path (.jsonl)
            output_path: Output directory path
            scene_path: Scene file path (.glb), if None, use scene_path from initialization or default scene in config file
            video_name: Custom video name, if None, automatically generated

        Returns:
            Tuple[str, dict]: (video file path, navigation statistics)

        Raises:
            ValueError: When no target points are detected or files do not exist
            FileNotFoundError: When input files do not exist
        """
        # Validate input files
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"Point cloud file does not exist: {ply_path}")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL file does not exist: {jsonl_path}")
        
        # Load data
        points = self._load_point_cloud(ply_path)
        pred_mask = self._load_segmentation_mask(jsonl_path)
        
        # Extract target points
        target_points = points[pred_mask == 1]
        if len(target_points) == 0:
            raise ValueError("No target points detected")
        
        target_center = target_points.mean(axis=0)
        target_center = self._transform_coordinates(target_center)
        
        # Determine the scene path to use
        used_scene_path = scene_path or self.scene_path
        if not used_scene_path:
            raise ValueError("Scene file path must be specified, either through scene_path parameter or scene_path parameter during initialization")
        
        # Validate if scene file exists
        if not os.path.exists(used_scene_path):
            raise FileNotFoundError(f"Scene file does not exist: {used_scene_path}")
        
        # Create configuration and environment
        config = self._create_habitat_config(used_scene_path)
        
        with habitat.Env(config=config) as env:
            # If no starting position specified, calculate based on scene boundaries
            if self.start_position is None:
                calculated_start_position = self._calculate_start_position(env)
            else:
                calculated_start_position = self.start_position.copy()
            
            # Navigation statistics
            nav_stats = {
                "target_points_count": len(target_points),
                "original_target_center": target_center.copy(),
                "start_position": calculated_start_position.copy(),
                "scene_path": used_scene_path
            }
            
            # Validate and adjust target coordinates
            target_center = self._validate_and_adjust_goal(target_center, env)
            nav_stats["adjusted_target_center"] = target_center.copy()
            
            # Create custom episode
            custom_episode = self._create_custom_episode(
                scene_path=used_scene_path,
                start_position=calculated_start_position,
                target_position=target_center.tolist()
            )
            
            # Manually set current episode
            env._current_episode = custom_episode
            
            # Target position in episode has been set during creation
            
            # Create agent
            agent = ShortestPathFollowerAgent(
                env=env,
                goal_radius=config.habitat.task.measurements.success.success_distance,
            )
            
            # Execute navigation
            observations = env.reset()
            agent.reset()
            
            vis_frames = []
            step_count = 0
            
            # Initial frame
            info = env.get_metrics()
            frame = observations_to_image(observations, info)
            info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            vis_frames.append(frame)
            
            # Navigation loop
            while not env.episode_over:
                action = agent.act(observations)
                if action is None:
                    break
                
                observations = env.step(action)
                info = env.get_metrics()
                frame = observations_to_image(observations, info)
                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)
                step_count += 1
            
            nav_stats["total_steps"] = step_count
            nav_stats["success"] = env.episode_over
            
            # Generate video
            if video_name is None:
                ply_filename = os.path.splitext(os.path.basename(ply_path))[0]
                scene_id = os.path.splitext(os.path.basename(custom_episode.scene_id))[0]
                video_name = f"{scene_id}_{custom_episode.episode_id}_{ply_filename}"
            
            os.makedirs(output_path, exist_ok=True)
            video_path = images_to_video(
                vis_frames, output_path, video_name, 
                fps=self.fps, quality=self.video_quality
            )
            
            nav_stats["video_path"] = video_path
            nav_stats["frames_count"] = len(vis_frames)
            
        return video_path, nav_stats
    
    def batch_navigate(self, 
                      tasks: List[Tuple[str, str, str, Optional[str]]], 
                      output_base_path: str) -> List[Tuple[str, dict]]:
        """
        Batch execute navigation tasks

        Args:
            tasks: Task list, each task is (ply_path, jsonl_path, task_name, scene_path)
            output_base_path: Output base path

        Returns:
            List[Tuple[str, dict]]: Result list for each task
        """
        results = []
        
        for i, task_info in enumerate(tasks):
            try:
                if len(task_info) == 3:
                    ply_path, jsonl_path, task_name = task_info
                    scene_path = None
                else:
                    ply_path, jsonl_path, task_name, scene_path = task_info
                    
                output_path = os.path.join(output_base_path, f"task_{i+1}_{task_name}")
                video_path, stats = self.navigate_to_target(
                    ply_path, jsonl_path, output_path, scene_path, task_name
                )
                results.append((video_path, stats))
                print(f"✅ Task {i+1}/{len(tasks)} completed: {task_name}")
            except Exception as e:
                print(f"❌ Task {i+1}/{len(tasks)} failed: {task_name}, error: {str(e)}")
                results.append((None, {"error": str(e)}))
        
        return results


# Convenience function
def quick_navigate(ply_path: str,
                  jsonl_path: str,
                  output_path: str,
                  scene_path: Optional[str] = None,
                  yaml_path: str = "config/benchmark/nav/pointnav/pointnav_scannet.yaml") -> str:
    """
    Quick navigation function

    Args:
        ply_path: Point cloud file path
        jsonl_path: Semantic segmentation result file path
        output_path: Output path
        scene_path: Scene file path (.glb)
        yaml_path: Habitat configuration file path

    Returns:
        str: Generated video file path
    """
    api = NavigationAPI(yaml_path=yaml_path, scene_path=scene_path)
    video_path, stats = api.navigate_to_target(ply_path, jsonl_path, output_path, scene_path)
    print(f"Navigation completed! Video saved to: {video_path}")
    print(f"Navigation statistics: {stats}")
    return video_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Habitat Navigation API")
    parser.add_argument("--ply_path", required=True, help="Point cloud file path")
    parser.add_argument("--jsonl_path", required=True, help="Semantic segmentation result file path")
    parser.add_argument("--output_path", required=True, help="Output directory path")
    parser.add_argument("--scene_path", help="Scene file path (.glb)")
    parser.add_argument("--yaml_path", 
                       default="config/benchmark/nav/pointnav/pointnav_scannet.yaml",
                       help="Habitat configuration file path")
    parser.add_argument("--video_name", help="Custom video name")
    
    args = parser.parse_args()
    
    try:
        api = NavigationAPI(yaml_path=args.yaml_path, scene_path=args.scene_path)
        video_path, stats = api.navigate_to_target(
            args.ply_path, 
            args.jsonl_path, 
            args.output_path,
            args.scene_path,
            args.video_name
        )
        print(f"✅ Navigation completed successfully!")
        print(f"📹 Video file: {video_path}")
        print(f"📊 Navigation statistics: {stats}")
    except Exception as e:
        print(f"❌ Navigation failed: {str(e)}")
