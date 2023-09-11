import argparse

import factor_graph
import helpers
import jax.numpy as jnp
import jaxlie
import numpy as np
from sample_generator import JointConnection
from tqdm import tqdm
from pathlib import Path

import time

import pandas as pd
import threading
import yaml

def get_SE3_pose(pos, ori):
    assert pos.shape == (3,)
    assert ori.shape == (4,)
    return jaxlie.SE3.from_rotation_and_translation(
        translation=jnp.array(pos), rotation=jaxlie.SO3.from_quaternion_xyzw(ori)
    )


def estimation_process(obs_tuple, idx, results_list, factor_graph_options, structure, joint_formulations):
    sub_poses_a, sub_poses_b = obs_tuple

    sub_poses_a = sub_poses_a #[::100]
    sub_poses_b = sub_poses_b #[::100]

    graph = factor_graph.graph.Graph()

    graph.build_graph(
        len(sub_poses_a),  # Amount of time steps
        structure,
        factor_graph_options,
        joint_formulations,
    )
    graph.update_poses({"first": sub_poses_a, 'second': sub_poses_b}, 1e-4)
    twist, base_transform, aux_data = graph.solve_graph(max_restarts=25,
                                                        aux_data_in={'parameters': None,
                                                                     'joint_states': None})

    results_list[idx] = {'twist': [float(x) for x in twist],
                         'base_tf': [float(x) for x in base_transform.wxyz_xyz],
                         'parameters': [float(x) for x in aux_data['parameters'].flatten()],
                         'joint_states': [float(x) for x in aux_data['joint_states'].flatten()]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", default=None, help="""Filepath""")
    args = parser.parse_args()

    data = pd.read_csv(args.file).to_numpy()

    poses_a  = data[:,1:8]
    poses_b  = data[:,8:-1]
    optimize = data[:,-1].astype(int)

    poses_a = zip(poses_a[:, :3], poses_a[:, 3:])
    poses_b = zip(poses_b[:, :3], poses_b[:, 3:])

    # eef_pos = data["eef_pos"]  # (T, 3)
    # eef_ori = data["eef_ori"]  # (T, 4); XYZW

    poses_a = [get_SE3_pose(pos, ori) for pos, ori in poses_a]
    poses_b = [get_SE3_pose(pos, ori) for pos, ori in poses_b]

    # eef_poses = [get_SE3_pose(pos, ori) for pos, ori in zip(eef_pos, eef_ori)]

    # Add 1 to include the marked observation and exclude it for the subsequent batch
    batch_idcs = np.vstack((np.hstack(([0], optimize.nonzero()[0][:-1] + 1)),
                            optimize.nonzero()[0] + 1)).T
                           

    obs_batches = [(poses_a[start:end], poses_b[start:end]) for start, end in batch_idcs]

    # Build the graph
    factor_graph_options = factor_graph.graph.GraphOptions(
        observe_transformation=False,
        observe_part_poses=True,
        observe_part_pose_betweens=False,
        observe_part_centers=False,
        seed_with_observations=False,
    )
    
    structure = {
        "first_second": JointConnection(
            from_id="first", to_id="second", via_id="first_second"
        )
    }

    joint_formulations = {
        "first_second": factor_graph.helpers.JointFormulation.GeneralTwist
    }

    results = [None] * len(obs_batches)

    # for sub_poses_a, sub_poses_b in tqdm(obs_batches, desc='Processing observations'):
    #     sub_poses_a = sub_poses_a[::100]
    #     sub_poses_b = sub_poses_b[::100]

    #     graph = factor_graph.graph.Graph()

    #     graph.build_graph(
    #         len(sub_poses_a),  # Amount of time steps
    #         structure,
    #         factor_graph_options,
    #         joint_formulations,
    #     )
    #     graph.update_poses({"first": sub_poses_a, 'second': sub_poses_b}, 1e-4)
    #     twist, base_transform, aux_data = graph.solve_graph(max_restarts=25)

    #     results.append({'twist': list(twist),
    #                     'base_tf': list(base_transform.wxyz_xyz)})

    #     print(twist)
    
    threads = []
    batch_idx = 0

    while batch_idx < len(obs_batches):
        if len(threads) < 10:
            threads.append(threading.Thread(target=estimation_process, args=(obs_batches[batch_idx], batch_idx, results, factor_graph_options, structure, joint_formulations)))
            threads[-1].start()
            batch_idx += 1
        else:
            time.sleep(1.0)
            for x in range(len(threads)):
                if not threads[x].is_alive():
                    del threads[x]
                    break
            

    for t in tqdm(threads, desc='Waiting for threads to finish'):
        t.join()

    with open(args.file[:-3] + 'yaml', 'w') as f:
        yaml.dump(results, f)
