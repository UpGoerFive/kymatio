import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


def plot_mst(scat_coeffs, J, L, src_img=None, label=None, cmap0='gray', cmap1='gray', cmap2='gray', log_r=False, r_is_scale=True, two_pi=False):
    '''
    Authors: Dr. Michael Glinsky and Francis Ogoke.

    Plots the first and second order transforms and prints oult the 0th order transform, and
    optionally the original image if it is given

    Parameters
    ----------
    scat_coeffs : numpy 3D array with indexes (MST_coeff_idx, MST_patch_i, MST_patch_j)
        Coefficients of the MST.
    J : int
        log2 of the samples in the patch, if J=log2(image) then there will be only one patch.
    L : int
        log2 of the number of angular sectors of the first order MST.
    src_img : numpy 2D array, optional
        image corresponding to the scat_coeffs. The default is None is plotted.
    label : str, optional
        label to title the plot. The default is None.
    cmap0 : str, optional
        colorbar to use for the original image. The default is 'gray'.
    cmap1 : str, optional
        colorbar to use for the 1st order MST. The default is 'gray'.
    cmap2 : str, optional
        colorbar to use for the 2nd order MST. The default is 'gray'.
    log_r : bool, optional
        plot a plot the radius logorithmically, otherwise linearly. The default is False.
    r_is_scale : bool, optional
        the radius is scale, otherwise it is 1/scale. The default is True.
    two_pi : bool, optional
        plot the second order sector from 0 to 2*pi, otherwise plot from 0 to pi. The default is False.

    Raises
    ------
    Exception
        If the dimensions of scat_coeffs do not match J and L.

    Returns
    -------
    fig : matplotlib figure object
        figure that was plotted of the MST.

    '''
    FONT_SIZE = 16

    # add option to plot colorbars
    print("coeffs shape: ", scat_coeffs.shape)

    len_order_0 = 1
    len_order_1 = J*L
    len_order_2 = (J*(J-1)//2)*(L**2)
    window_rows, window_columns = scat_coeffs.shape[1:]
    print("number of (order 0, order 1, order 2) coefficients: ", (len_order_0, len_order_1, len_order_2))
    print("number of window rows and columns: ", (window_rows, window_columns))

    length_mst = len_order_0 + len_order_1 + len_order_2
    if not scat_coeffs.shape[0] == length_mst:
        raise Exception('The value of L=%s and J=%s imply length of MST=%s, but it is %s' % (str(L),str(J),str(length_mst),str(scat_coeffs.shape[0])))

    ####################################################################
    # We now retrieve zeroth, first-order and second-order coefficients for the display.
    scat_coeffs_order_0 = scat_coeffs[0,:,:]

    scat_coeffs_order_1 = scat_coeffs[1:1+len_order_1, :, :]
    norm_order_1 = mpl.colors.Normalize(scat_coeffs_order_1.min(), scat_coeffs_order_1.max(), clip=True)
    mapper_order_1 = cm.ScalarMappable(norm=norm_order_1, cmap=cmap1)
    # Mapper of coefficient amplitude to a grayscale color for visualisation.

    scat_coeffs_order_2 = scat_coeffs[1+len_order_1:, :, :]
    norm_order_2 = mpl.colors.Normalize(scat_coeffs_order_2.min(), scat_coeffs_order_2.max(), clip=True)
    mapper_order_2 = cm.ScalarMappable(norm=norm_order_2, cmap=cmap2)
    # Mapper of coefficient amplitude to a grayscale color for visualisation.

    # print out values of zeroth order coefficient
    print('MST of order 0 = %s' % (str(scat_coeffs_order_0)))

    if not src_img is None:
        fig = plt.figure(figsize=(19.2, 6))
        spec = fig.add_gridspec(ncols=3, nrows=1)
        gs = gridspec.GridSpec(1, 3, wspace=0.1)
        gs_order_1 = gridspec.GridSpecFromSubplotSpec(window_rows, window_columns, subplot_spec=gs[1])
        gs_order_2 = gridspec.GridSpecFromSubplotSpec(window_rows, window_columns, subplot_spec=gs[2])
    else:
        fig = plt.figure(figsize=(12.8, 6))
        spec = fig.add_gridspec(ncols=2, nrows=1)
        gs = gridspec.GridSpec(1, 2, wspace=0.1)
        gs_order_1 = gridspec.GridSpecFromSubplotSpec(window_rows, window_columns, subplot_spec=gs[0])
        gs_order_2 = gridspec.GridSpecFromSubplotSpec(window_rows, window_columns, subplot_spec=gs[1])

    # Start by plotting input
    if not src_img is None:
        ax = plt.subplot(gs[0])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(src_img,cmap=cmap0,interpolation='nearest', aspect='auto')
        ax.axis('off')
        if label:
            ax.set_title(label, fontsize=FONT_SIZE)

    # Plot first-order scattering coefficients
    if not src_img is None:
        ax = plt.subplot(gs[1])
    else:
        ax = plt.subplot(gs[0])
    ax.set_xticks([])
    ax.set_yticks([])
    if not label is None:
        ax.set_title('MST_1( ' + label + ' )', fontsize=FONT_SIZE)

    l_offset = int(L - L / 2 - 1)  # follow same ordering as Kymatio for angles

    for row in range(window_rows):
        for column in range(window_columns):
            ax = fig.add_subplot(gs_order_1[row, column], projection='polar')
            ax.axis('off')
            coefficients = scat_coeffs_order_1[:, row, column]
            for j in range(J):
                if r_is_scale:
                    if log_r:
                        height = 1
                        bottom = j
                    else:
                        if j == 0:
                            height = 4
                            bottom = 0
                        else:
                            height = 2 ** (j + 1)
                            bottom = height
                else:
                    if log_r:
                        height = 1
                        bottom = J - 1 - j
                    else:
                        if j == J-1:
                            height = 2 ** (- J + 1.0)
                            bottom = 0
                        else:
                            height = 2 ** (- j - 1)
                            bottom = height
                # print('plotting sector = ', j, bottom, bottom + height)
                for l in range(L):
                    coeff = coefficients[l + j * L]
                    color = mapper_order_1.to_rgba(coeff)
                    angle = (l_offset - l) * np.pi / L
                    ax.bar(x=angle,
                           height=height,
                           width=np.pi / L,
                           bottom=bottom,
                           edgecolor=color,
                           color=color)
                    ax.bar(x=angle + np.pi,
                           height=height,
                           width=np.pi / L,
                           bottom=bottom,
                           edgecolor=color,
                           color=color)

    # Plot second-order scattering coefficients
    if not src_img is None:
        ax = plt.subplot(gs[2])
    else:
        ax = plt.subplot(gs[1])
    ax.set_xticks([])
    ax.set_yticks([])
    if not label is None:
        ax.set_title('MST_2( ' + label + ' )', fontsize=FONT_SIZE)

    for row in range(window_rows):
        for column in range(window_columns):
            ax = fig.add_subplot(gs_order_2[row, column], projection='polar')
            ax.axis('off')
            coefficients = scat_coeffs_order_2[:, row, column]
            for j1 in range(J - 1):
                for j2 in range(j1 + 1, J):
                    if r_is_scale:
                        if log_r:
                            height = 1.0 / j2
                            bottom = (j2-1) + j1 / j2
                        else:
                            if j1 == 0:
                                height = 4
                                bottom = 2 ** (j2 + 1)
                            else:
                                height = 2 ** (j1 + 1)
                                bottom = 2 ** (j2 + 1) + height
                    else:
                        if log_r:
                            height = 1.0 / (J - j1 - 1)
                            bottom = (J - j1 - 2) + (J - j2 - 1) / (J - j1 - 1)
                        else:
                            if j2 == J-1:
                                height = 2 ** (- J + 1.0)
                                bottom = 2 ** (- j1 - 1)
                            else:
                                height = 2 ** (- j2 - 1)
                                bottom = 2 ** (- j1 - 1) + height
                    # print('plotting sector = ', j1, j2, bottom, bottom + height)
                    for l1 in range(L):
                        for l2 in range(L):
                            coeff_index = l1 * L * (J - j1 - 1) + l2 + L * (j2 - j1 - 1) + (L ** 2) * \
                                          (j1 * (J - 1) - j1 * (j1 - 1) // 2)
                            # indexing a bit complex which follows the order used by Kymatio to compute
                            # scattering coefficients
                            coeff = coefficients[coeff_index]
                            color = mapper_order_2.to_rgba(coeff)
                            # split along angles first-order quadrants in L quadrants, using same ordering
                            # as Kymatio (clockwise) and center (with the 0.5 offset)
                            if two_pi:
                                angle = (l_offset - l1) * np.pi / L + 0.5 * (L // 2 - l2 - 0.5) * np.pi / (L ** 2)
                                angle2 = (l_offset - l1) * np.pi / L + 0.5 * (L // 2 + l2 + 0.5) * np.pi / (L ** 2)
                                angle3 = (l_offset - l1 - 1) * np.pi / L + 0.5 * (L // 2 + l2 + 0.5) * np.pi / (L ** 2)
                            else:
                                angle = (l_offset - l1) * np.pi / L + (L // 2 - l2 - 0.5) * np.pi / (L ** 2)
                            # equal split along radius is performed through height variable
                            if two_pi:
                                ax.bar(x=angle,
                                       height=height,
                                       width= 0.5 * np.pi / L ** 2,
                                       bottom=bottom,
                                       edgecolor=color,
                                       color=color)
                                ax.bar(x=angle + np.pi,
                                       height=height,
                                       width= 0.5 *np.pi / L ** 2,
                                       bottom=bottom,
                                       edgecolor=color,
                                       color=color)
                                if l2 < L // 2:
                                    ax.bar(x=angle2,
                                            height=height,
                                            width= 0.5 * np.pi / L ** 2,
                                            bottom=bottom,
                                            edgecolor=color,
                                            color=color)
                                    ax.bar(x=angle2 + np.pi,
                                            height=height,
                                            width= 0.5 * np.pi / L ** 2,
                                            bottom=bottom,
                                            edgecolor=color,
                                            color=color)
                                else:
                                    ax.bar(x=angle3,
                                            height=height,
                                            width= 0.5 * np.pi / L ** 2,
                                            bottom=bottom,
                                            edgecolor=color,
                                            color=color)
                                    ax.bar(x=angle3 + np.pi,
                                            height=height,
                                            width= 0.5 * np.pi / L ** 2,
                                            bottom=bottom,
                                            edgecolor=color,
                                            color=color)
                            else:
                                ax.bar(x=angle,
                                       height=height,
                                       width= np.pi / L ** 2,
                                       bottom=bottom,
                                       edgecolor=color,
                                       color=color)
                                ax.bar(x=angle + np.pi,
                                       height=height,
                                       width= np.pi / L ** 2,
                                       bottom=bottom,
                                       edgecolor=color,
                                       color=color)
    # return fig


def refactor_plot_mst(scat_coeffs, J, L, src_img=None, label=None, cmap0='gray', cmap1='gray', cmap2='gray', log_r=False, r_is_scale=True, two_pi=False):
    '''
    Authors: Dr. Michael Glinsky and Francis Ogoke.

    Plots the first and second order transforms and prints oult the 0th order transform, and
    optionally the original image if it is given

    Parameters
    ----------
    scat_coeffs : numpy 3D array with indexes (MST_coeff_idx, MST_patch_i, MST_patch_j)
        Coefficients of the MST.
    J : int
        log2 of the samples in the patch, if J=log2(image) then there will be only one patch.
    L : int
        log2 of the number of angular sectors of the first order MST.
    src_img : numpy 2D array, optional
        image corresponding to the scat_coeffs. The default is None is plotted.
    label : str, optional
        label to title the plot. The default is None.
    cmap0 : str, optional
        colorbar to use for the original image. The default is 'gray'.
    cmap1 : str, optional
        colorbar to use for the 1st order MST. The default is 'gray'.
    cmap2 : str, optional
        colorbar to use for the 2nd order MST. The default is 'gray'.
    log_r : bool, optional
        plot a plot the radius logorithmically, otherwise linearly. The default is False.
    r_is_scale : bool, optional
        the radius is scale, otherwise it is 1/scale. The default is True.
    two_pi : bool, optional
        plot the second order sector from 0 to 2*pi, otherwise plot from 0 to pi. The default is False.

    Raises
    ------
    Exception
        If the dimensions of scat_coeffs do not match J and L.

    Returns
    -------
    fig : matplotlib figure object
        figure that was plotted of the MST.

    '''
    FONT_SIZE = 16

    # add option to plot colorbars
    print("coeffs shape: ", scat_coeffs.shape)

    len_order_0 = 1
    len_order_1 = J*L
    len_order_2 = (J*(J-1)//2)*(L**2)
    window_rows, window_columns = scat_coeffs.shape[1:]
    print("number of (order 0, order 1, order 2) coefficients: ", (len_order_0, len_order_1, len_order_2))
    print("number of window rows and columns: ", (window_rows, window_columns))

    length_mst = len_order_0 + len_order_1 + len_order_2
    if not scat_coeffs.shape[0] == length_mst:
        raise Exception('The value of L=%s and J=%s imply length of MST=%s, but it is %s' % (str(L),str(J),str(length_mst),str(scat_coeffs.shape[0])))

    ####################################################################
    # We now retrieve zeroth, first-order and second-order coefficients for the display.
    scat_coeffs_order_0 = scat_coeffs[0,:,:]

    scat_coeffs_order_1 = scat_coeffs[1:1+len_order_1, :, :]
    norm_order_1 = mpl.colors.Normalize(scat_coeffs_order_1.min(), scat_coeffs_order_1.max(), clip=True)
    mapper_order_1 = cm.ScalarMappable(norm=norm_order_1, cmap=cmap1)
    # Mapper of coefficient amplitude to a grayscale color for visualisation.

    scat_coeffs_order_2 = scat_coeffs[1+len_order_1:, :, :]
    norm_order_2 = mpl.colors.Normalize(scat_coeffs_order_2.min(), scat_coeffs_order_2.max(), clip=True)
    mapper_order_2 = cm.ScalarMappable(norm=norm_order_2, cmap=cmap2)
    # Mapper of coefficient amplitude to a grayscale color for visualisation.

    # print out values of zeroth order coefficient
    print('MST of order 0 = %s' % (str(scat_coeffs_order_0)))

    if not src_img is None:
        fig = plt.figure(figsize=(19.2, 6))
        spec = fig.add_gridspec(ncols=3, nrows=1)
        gs = gridspec.GridSpec(1, 3, wspace=0.1)
        gs_order_1 = gridspec.GridSpecFromSubplotSpec(window_rows, window_columns, subplot_spec=gs[1])
        gs_order_2 = gridspec.GridSpecFromSubplotSpec(window_rows, window_columns, subplot_spec=gs[2])
    else:
        fig = plt.figure(figsize=(12.8, 6))
        spec = fig.add_gridspec(ncols=2, nrows=1)
        gs = gridspec.GridSpec(1, 2, wspace=0.1)
        gs_order_1 = gridspec.GridSpecFromSubplotSpec(window_rows, window_columns, subplot_spec=gs[0])
        gs_order_2 = gridspec.GridSpecFromSubplotSpec(window_rows, window_columns, subplot_spec=gs[1])

    # Start by plotting input
    if not src_img is None:
        ax = plt.subplot(gs[0])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(src_img,cmap=cmap0,interpolation='nearest', aspect='auto')
        ax.axis('off')
        if label:
            ax.set_title(label, fontsize=FONT_SIZE)

    # Plot first-order scattering coefficients
    if not src_img is None:
        ax = plt.subplot(gs[1])
    else:
        ax = plt.subplot(gs[0])
    ax.set_xticks([])
    ax.set_yticks([])
    if not label is None:
        ax.set_title('MST_1( ' + label + ' )', fontsize=FONT_SIZE)

    l_offset = int(L - L / 2 - 1)  # follow same ordering as Kymatio for angles

    for row in range(window_rows):
        for column in range(window_columns):
            ax = fig.add_subplot(gs_order_1[row, column], projection='polar')
            ax.axis('off')
            coefficients = scat_coeffs_order_1[:, row, column]
            heights = []
            bottoms = []
            angles = []
            coeff_colors = []
            for j in range(J):
                if r_is_scale:
                    if log_r:
                        height = 1
                        bottom = j
                    else:
                        if j == 0:
                            height = 4
                            bottom = 0
                        else:
                            height = 2 ** (j + 1)
                            bottom = height
                else:
                    if log_r:
                        height = 1
                        bottom = J - 1 - j
                    else:
                        if j == J-1:
                            height = 2 ** (- J + 1.0)
                            bottom = 0
                        else:
                            height = 2 ** (- j - 1)
                            bottom = height
                # print('plotting sector = ', j, bottom, bottom + height)
                for l in range(L):
                    heights.append(height)
                    bottoms.append(bottom)
                    coeff = coefficients[l + j * L]
                    color = mapper_order_1.to_rgba(coeff)
                    coeff_colors.append(color)
                    angle = (l_offset - l) * np.pi / L
                    angles.append(angle)
            ax.bar(x=angles,
                    height=heights,
                    width=np.pi / L,
                    bottom=bottoms,
                    edgecolor=coeff_colors,
                    color=coeff_colors)
            ax.bar(x=np.array(angles) + np.pi,
                    height=heights,
                    width=np.pi / L,
                    bottom=bottoms,
                    edgecolor=coeff_colors,
                    color=coeff_colors)

    # Plot second-order scattering coefficients
    if not src_img is None:
        ax = plt.subplot(gs[2])
    else:
        ax = plt.subplot(gs[1])
    ax.set_xticks([])
    ax.set_yticks([])
    if not label is None:
        ax.set_title('MST_2( ' + label + ' )', fontsize=FONT_SIZE)

    for row in range(window_rows):
        for column in range(window_columns):
            ax = fig.add_subplot(gs_order_2[row, column], projection='polar')
            ax.axis('off')
            coefficients = scat_coeffs_order_2[:, row, column]
            heights = []
            bottoms = []
            angles = []
            coeff_colors = []
            for j1 in range(J - 1):
                for j2 in range(j1 + 1, J):
                    if r_is_scale:
                        if log_r:
                            height = 1.0 / j2
                            bottom = (j2-1) + j1 / j2
                        else:
                            if j1 == 0:
                                height = 4
                                bottom = 2 ** (j2 + 1)
                            else:
                                height = 2 ** (j1 + 1)
                                bottom = 2 ** (j2 + 1) + height
                    else:
                        if log_r:
                            height = 1.0 / (J - j1 - 1)
                            bottom = (J - j1 - 2) + (J - j2 - 1) / (J - j1 - 1)
                        else:
                            if j2 == J-1:
                                height = 2 ** (- J + 1.0)
                                bottom = 2 ** (- j1 - 1)
                            else:
                                height = 2 ** (- j2 - 1)
                                bottom = 2 ** (- j1 - 1) + height
                    # print('plotting sector = ', j1, j2, bottom, bottom + height)
                    for l1 in range(L):
                        for l2 in range(L):
                            heights.append(height)
                            bottoms.append(bottom)
                            coeff_index = l1 * L * (J - j1 - 1) + l2 + L * (j2 - j1 - 1) + (L ** 2) * \
                                          (j1 * (J - 1) - j1 * (j1 - 1) // 2)
                            # indexing a bit complex which follows the order used by Kymatio to compute
                            # scattering coefficients
                            coeff = coefficients[coeff_index]
                            coeff_colors.append(mapper_order_2.to_rgba(coeff))
                            # split along angles first-order quadrants in L quadrants, using same ordering
                            # as Kymatio (clockwise) and center (with the 0.5 offset)

                            # equal split along radius is performed through height variable
                            if two_pi:
                                angles.append((l_offset - l1) * np.pi / L + 0.5 * (L // 2 - l2 - 0.5) * np.pi / (L ** 2))
                                width= 0.5 * np.pi / L ** 2,
                                if l2 < L // 2:
                                    angles.append((l_offset - l1) * np.pi / L + 0.5 * (L // 2 + l2 + 0.5) * np.pi / (L ** 2))
                                    heights.append(height)
                                    bottoms.append(bottom)
                                else:
                                    angles.append((l_offset - l1 - 1) * np.pi / L + 0.5 * (L // 2 + l2 + 0.5) * np.pi / (L ** 2))
                                    heights.append(height)
                                    bottoms.append(bottom)
                            else:
                                angles.append((l_offset - l1) * np.pi / L + (L // 2 - l2 - 0.5) * np.pi / (L ** 2))
                                width= np.pi / L ** 2,
            ax.bar(x=angles,
                    height=heights,
                    width=width,
                    bottom=bottoms,
                    edgecolor=coeff_colors,
                    color=coeff_colors)
            ax.bar(x=np.array(angles) + np.pi,
                    height=heights,
                    width=width,
                    bottom=bottoms,
                    edgecolor=coeff_colors,
                    color=coeff_colors)
    # return fig


def scatnet_to_Kymatio_reorder(array, J, L):
    """
    Author: Nathaniel Martin
    Reorders a scatnet ordered coefficient array to Kymatio ordering, in other words, reverses the L ordering.
    Max order can be 1 or 2.
    """
    len_order_1 = J*L
    len_order_2 = (J*(J-1)//2)*(L**2)
    ordering_length = (len_order_2 //L)

    first_ordering = np.array([l + j * L for j in range(J) for l in range(L, 0, -1)])

    small_loop_ordering = np.array([l + j * L for j in range(ordering_length) for l in range(L-1,-1,-1)])
    second_layer_ordering = []
    past_idx = 0
    for j in range(J-1, -1, -1):
        # length of j * L * L, we've already done a little L reversal so all we need to do is reverse the big loop
        start_chunk = small_loop_ordering[past_idx:past_idx+j*L**2]
        past_idx += j*L**2
        reversed_chunk = []
        for l in range(L, 0, -1):
            reversed_chunk.extend(start_chunk[(l-1)*j*L:l*j*L])
        second_layer_ordering.append(reversed_chunk)

    second_ordering = np.concatenate(second_layer_ordering).astype(int)

    second_ordering = second_ordering + 1 + len_order_1
    total_order = np.concatenate((np.array([0]), first_ordering, second_ordering))

    return array[total_order]