import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import connected_components
from sklearn.mixture import GaussianMixture


def get_diameter_dist_correspondence():
    # diameter in pixels to distance from base offset in mm (without scintillator)
    diam_to_dist = {}
    diam_to_dist[25] = -5.1
    diam_to_dist[30] = -3.8
    diam_to_dist[35] = -2.5
    diam_to_dist[45] = -1.3
    diam_to_dist[55] = 0.0 # corresponds to 39 mm away from lens (apparent bottom of scintillator)
    diam_to_dist[60] = 1.3
    diam_to_dist[70] = 2.5
    diam_to_dist[85] = 3.8
    diam_to_dist[95] = 5.1
    diam_to_dist[110] = 6.3
    diam_to_dist[125] = 7.6
    diam_to_dist[145] = 8.9
    diam_to_dist[165] = 10.2
    diam_to_dist[185] = 11.4
    diam_to_dist[210] = 12.7
    diam_to_dist[245] = 14.0
    diam_to_dist[270] = 15.2
    return diam_to_dist


# minimum spanning tree
def denoise_MST(img, T_edge, int_min_count, num_ints=1):
    img_denoised = np.copy(img)
    num_nodes = int(img.sum())
    node_to_y_x = {}
    node_locs = np.argwhere(img)
    # distance from a node to all other nodes, for each node
    graph = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        node_to_y_x[i] = node_locs[i]
        for j in range(num_nodes):
            if j != i:
                graph[i,j] = np.linalg.norm(node_locs[i] - node_locs[j])
    X = csr_matrix(graph)
    Tcsr = minimum_spanning_tree(X)
    graph_mst = Tcsr.toarray()
    graph_mst_denoised = np.copy(graph_mst)
    # remove edges that are longer than a threshold
    graph_mst_denoised[graph_mst_denoised > T_edge] = 0
    # make graph symmetric (undirected)
    graph_mst_denoised = np.maximum(graph_mst_denoised, graph_mst_denoised.T)
    for i in range(num_nodes):
        if graph_mst_denoised[i].sum() == 0:
            img_denoised[node_to_y_x[i][0], node_to_y_x[i][1]] = 0
    n_components, labels = connected_components(csr_matrix(graph_mst_denoised), directed=False)
    # get id of largest connected component (most frequent label)
    largest_comp_id = np.argmax(np.bincount(labels))
    largest_comp_count = np.max(np.bincount(labels))
    if largest_comp_count < int_min_count:
        return None
    if num_ints > 1:
        other_labels = labels[labels!=largest_comp_id]
        if len(other_labels) == 0:
            return None
        second_largest_comp_id = np.argmax(np.bincount(other_labels))
        second_largest_comp_count = np.max(np.bincount(other_labels))
        # return empty frame if 2nd interaction did not have minimum number of counts required
        if second_largest_comp_count < int_min_count:
            return None
    # remove photons not in the largest connected component
    for i in range(num_nodes):
        lab = labels[i]
        if num_ints == 1:
            if lab != largest_comp_id:
                img_denoised[node_to_y_x[i][0], node_to_y_x[i][1]] = 0
        else:
            if lab != largest_comp_id and lab != second_largest_comp_id:
                img_denoised[node_to_y_x[i][0], node_to_y_x[i][1]] = 0
    return img_denoised


# GMM-Loc
def pred_ints_gmm(frame, num_ints, s):
    pix = np.argwhere(frame) # n x 2
    gm = GaussianMixture(n_components=num_ints, covariance_type="spherical", random_state=0, means_init=None).fit(pix)
    centroids = gm.means_ # centroids: num_ints x (y,x)
    diams = s*gm.covariances_**.5
    ints_num_photons = np.zeros((num_ints,))
    assigned_to_int = np.zeros(pix.shape[0], dtype=bool)
    for i in range(num_ints):
        # assign photons contained within circle to corresponding interaction
        photon_dist_from_circle = np.linalg.norm(pix - centroids[i], axis=1) - diams[i]/2
        int_mask = photon_dist_from_circle < 0
        assigned_to_int[int_mask] = True
        ints_num_photons[i] = int_mask.sum()
    unassigned_photons = pix[~assigned_to_int]
    for i in range(unassigned_photons.shape[0]):
        # assign remaining photons (outside from circles) to closest circle
        phot_loc = unassigned_photons[i].reshape(1,2)
        int_choice = np.argmin(np.linalg.norm(phot_loc - centroids, axis=1) - diams/2)
        ints_num_photons[int_choice] += 1
    return centroids, diams.reshape(-1,1), ints_num_photons.reshape(-1,1)


def convert_centroid_diam_to_world(centroids, diams, diam_to_dist_dict, lens_to_sensor_dist):
    # SPAD array pixels
    num_y_pixels = 256
    num_x_pixels = 496
    dx = 9.5 / 511
    dy = 9.6 / 511

    n = 1.79
    d = 2.7
    S2 = lens_to_sensor_dist
    diams_ra = np.array(list(diam_to_dist_dict.keys()))
    dists_ra = np.array(list(diam_to_dist_dict.values())) # distance from bottom of scint in mm
    num_ints = len(diams)
    pred_locs = np.zeros((num_ints, 3))
    for i in range(num_ints):
        dist_from_bot = np.interp(diams[i], diams_ra, dists_ra)
        dist_from_top = 39 - dist_from_bot
        S3 = dist_from_top # distance from lens mount edge to apparent interaction depth
        S4 = d + (S3 - d) * n
        x_image = (centroids[i,1] - num_x_pixels/2) * dx
        y_image = (centroids[i,0] - num_y_pixels/2) * dy
        pred_x = -x_image * S3 / S2
        pred_y = -y_image * S3 / S2
        pred_z = 70 + d - S4 # scintillator is 70 mm in length
        pred_locs[i] = np.array([pred_x, pred_y, pred_z]).reshape(1,3)
    return pred_locs


def localize_frames(frames_ra):
    num_frames = frames_ra.shape[0]
    diam_to_dist_dict = get_diameter_dist_correspondence()
    lens_to_sensor_dist = 58 # S_2
    T_edge = 46
    min_counts = 5 # discard denoised frame if below minimum counts threshold
    s = 3.7 # GMM parameter
    num_ints = 1 # number of interactions to locate
    pred_loc_ra = []
    for i in range(num_frames):
        frame = frames_ra[i]
        frame_denoised = denoise_MST(frame, T_edge, min_counts, num_ints)
        if frame_denoised is None:
            continue
        centroids_pix, diams_pix, ints_num_photons = pred_ints_gmm(frame_denoised, num_ints, s)
        pred_loc = convert_centroid_diam_to_world(centroids_pix, diams_pix, diam_to_dist_dict, lens_to_sensor_dist)
        pred_loc_ra.append(pred_loc)
    pred_loc_ra = np.concatenate(pred_loc_ra, axis=0)
    return pred_loc_ra


if __name__ == "__main__":
    fname = "x-y_sensitivity/x=-12mm.npy"
    # fname = "x-y_sensitivity/x=0mm.npy"
    # fname = "x-y_sensitivity/x=12mm.npy"
    # fname = "z_sensitivity/z=0mm.npy"
    # fname = "z_sensitivity/z=10mm.npy"
    # fname = "z_sensitivity/z=20mm.npy"
    frames_ra = np.load(fname)
    pred_loc_ra = localize_frames(frames_ra)
    print(pred_loc_ra.mean(axis=0))




