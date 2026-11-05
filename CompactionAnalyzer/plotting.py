import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib_scalebar.scalebar import ScaleBar
import warnings
from pathlib import Path
#matplotlib.pyplot.switch_backend("agg")


# Explicit axis-angle names shared by the full and reduced analysis modes.
# "min" is the FIBER direction; "max" is the perpendicular GRADIENT direction.
# Both vectors retain their existing components. The stored
# vector components are [x, y_down], despite the historical ax0/ax1 filenames.
# Angle files are derived representations of these unit eigenvectors, not
# additional measurements. Only fiber x/y angles are exported; separate
# maximum-eigenvector angle files would duplicate these directions.
AXIS_ARRAY_FILENAMES = {
    "min_x": "Angles_towards_x_axis.npy",
    "min_y": "Angles_towards_y_axis.npy",
    "min_vx": "Vector_min_ax0.npy",
    "min_vy": "Vector_min_ax1.npy",
    "max_vx": "Vector_max_ax0.npy",
    "max_vy": "Vector_max_ax1.npy",
    "coherency": "CoherencyMap.npy",
    "fiber": "FiberImageCrop.npy",
}
# Read compatibility only: these files may exist in earlier result folders.
LEGACY_MAX_ARRAY_FILENAMES = {
    "max_x": "Angles_max_evec_towards_x_axis.npy",
    "max_y": "Angles_max_evec_towards_y_axis.npy",
}
LEGACY_ANGLE_FILENAME = "AngleMap_NoReference.npy"


def wrap_axis_angle(angle):
    """Axial angles in degrees: theta and theta + 180 describe the same fiber."""
    return (np.asarray(angle) + 90.0) % 180.0 - 90.0


def line_reference_angle(reference_line, shape=None):
    """Axial angle of two distinct (x, y_down) points, in degrees.

    The reference is an infinite line, not a segment or radial origin.
    Reversing the endpoints gives the same axis. Optional shape validates
    uncropped pixel coordinates against (height, width).
    """
    message = 'reference_line must contain two distinct finite (x, y) points inside the image'
    try:
        points = np.asarray(reference_line, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError(message)
    if shape is not None and (np.any(points < 0) or np.any(points[:, 0] >= shape[1])
                              or np.any(points[:, 1] >= shape[0])):
        raise ValueError(message)
    dx, dy = points[1] - points[0]
    if np.hypot(dx, dy) <= 1e-12:
        raise ValueError(message)
    return float(wrap_axis_angle(np.degrees(np.arctan2(-dy, dx))))


def convert_axis_reference(angle_x, reference="x"):
    """Convert the saved x angle; this is NOT a gradient-to-fiber correction.

    x: horizontal=0, vertical=-90; +45 slopes upwards to the right in an image.
    y: signed angle relative to the y axis, with the SAME sense of rotation.
    legacy_flip: EXACT former reader behavior 90 - angle_x, including +90 at
                 angle_x=0. It is kept explicitly for reproducing old results.
    Image rows point down; positive mathematical angles point upwards.
    """
    angle_x = np.asarray(angle_x)
    if reference == "x":
        return angle_x.copy()
    if reference == "y":
        return wrap_axis_angle(angle_x - 90.0)
    if reference == "legacy_flip":
        flipped = 90.0 - angle_x
        return np.where(flipped > 90, flipped - 180,
                        np.where(flipped < -90, flipped + 180, flipped))
    raise ValueError("reference must be 'x', 'y', or 'legacy_flip'")


def vectors_to_axis_angles(vectors):
    """Return (x-angle, y-angle) from eigenvectors with components [x, y_down].

    The x formula is intentionally identical to the existing full-mode
    AngleMap_NoReference calculation. Both modes and saved-data plots use this
    helper, so changing the plotting reference cannot silently change analysis.
    Nonfinite/zero vectors have undefined orientation and remain NaN.
    """
    vectors = np.asarray(vectors)
    if vectors.ndim != 3 or vectors.shape[-1] != 2:
        raise ValueError("Expected an (height, width, 2) eigenvector array")
    vx, vy = vectors[..., 0], vectors[..., 1]
    angle_x = (np.arctan2(vx, vy) * 360 / (2 * np.pi)) % 180 - 90
    valid = np.isfinite(vx) & np.isfinite(vy) & (np.hypot(vx, vy) > 0)
    angle_x = np.where(valid, angle_x, np.nan)
    return angle_x, convert_axis_reference(angle_x, "y")


def load_axis_arrays(numpy_folder):
    """Load current .npy maps or older full-mode results, without pickle.

    Old AngleMap_NoReference is already an x-axis FIBER angle: never apply the
    former flip implicitly. Missing y maps are derived from x. Max-eigenvector
    angles from earlier exports are still read/derived when present. New exports
    retain both vectors but save explicit angle maps only for the fiber direction.
    """
    folder = Path(numpy_folder)
    arrays = {key: np.load(folder / filename, allow_pickle=False)
              for key, filename in {**AXIS_ARRAY_FILENAMES, **LEGACY_MAX_ARRAY_FILENAMES}.items()
              if (folder / filename).is_file()}
    legacy = folder / LEGACY_ANGLE_FILENAME
    if legacy.is_file():
        legacy_x = np.load(legacy, allow_pickle=False)
        # Reusing a folder with an older package can leave stale new filenames.
        # Both readers must see the same x angles, including undefined pixels.
        if "min_x" in arrays and (arrays["min_x"].shape != legacy_x.shape or
                not np.allclose(arrays["min_x"], legacy_x, rtol=0, atol=1e-5, equal_nan=True)):
            raise ValueError(f"Inconsistent old/new x-angle maps; use separate output folders: {folder}")
        if "min_x" not in arrays:
            arrays["min_x"] = legacy_x
    shapes = {np.shape(value) for value in arrays.values()}
    if len(shapes) > 1 or any(len(shape) != 2 for shape in shapes):
        raise ValueError(f"Orientation files must be 2D arrays with identical shapes: {folder}")
    for eigenvector in ("min", "max"):
        x_key, y_key = f"{eigenvector}_x", f"{eigenvector}_y"
        if x_key not in arrays and all(f"{eigenvector}_v{a}" in arrays for a in ("x", "y")):
            vector = np.stack([arrays[f"{eigenvector}_vx"], arrays[f"{eigenvector}_vy"]], axis=-1)
            arrays[x_key], derived_y = vectors_to_axis_angles(vector)
        elif x_key in arrays:
            derived_y = convert_axis_reference(arrays[x_key], "y")
        else:
            continue
        if y_key in arrays:
            matching_nan = np.array_equal(np.isnan(arrays[y_key]), np.isnan(derived_y))
            matching_angle = np.allclose(wrap_axis_angle(arrays[y_key] - derived_y),
                np.where(np.isnan(derived_y), np.nan, 0), equal_nan=True, atol=1e-5)
            if not matching_nan or not matching_angle:
                raise ValueError(f"Inconsistent x/y angle maps for {eigenvector}: {folder}")
        arrays[y_key] = derived_y if y_key not in arrays else arrays[y_key]
    if "min_x" not in arrays:
        raise FileNotFoundError(f"No fiber angle or minimum-eigenvector maps found in {folder}")
    mask_file = folder / "ValidityMask.npy"
    if mask_file.is_file():
        mask = np.load(mask_file, allow_pickle=False)
        if mask.dtype != bool or mask.shape != arrays["min_x"].shape:
            raise ValueError("ValidityMask.npy must be a boolean array matching the angle maps")
        arrays["validity_mask"] = mask
    return arrays


def plot_axis_orientation_arrays(arrays, output_folder, scale=None, dpi=200,
                                 mask=None, cell=None, include_coherency=True):
    """Use the existing package overlay style for the fiber axis maps.

    During analysis, fiber is the once-normalized image used by the old plots.
    A saved FiberImageCrop.npy was normalized a second time, so a later replot
    from that array can have different background contrast. Angles are equal.
    The optional mask changes display only; it never changes saved arrays.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    shape = arrays["min_x"].shape
    if mask is None:
        mask = arrays.get("validity_mask")
    for name, value in (("mask", mask), ("cell", cell), ("fiber", arrays.get("fiber"))):
        if value is not None and np.shape(value) != shape:
            raise ValueError(f"{name} shape does not match the angle maps")
    fields = []
    for axis in ("x", "y"):
        key = f"min_{axis}"
        if key in arrays:
            fields.append((key, f"Angle ({axis}-axis)", -90, 90, "gist_rainbow"))
    if include_coherency and "coherency" in arrays:
        fields.append(("coherency", "Coherency", 0, 1, "turbo"))
    paths = []
    for key, label, vmin, vmax, cmap in fields:
        field = arrays[key]
        if mask is not None:
            field = np.where(np.asarray(mask, dtype=bool), field, np.nan)
        name = Path(AXIS_ARRAY_FILENAMES[key]).with_suffix(".png").name
        path = output_folder / name
        # Reuse the original renderer, including alpha, layer order, colorbar,
        # figure size and scalebar. No separate new plotting style is needed.
        fig = plot_fancy_overlay(arrays.get("fiber", np.zeros(shape)), cell, field,
            path_png=str(path), label=label, dpi=dpi, cmap_angle=cmap,
            omin=vmin, omax=vmax, scale=scale)
        paths.append(path)
        plt.close(fig)
    return paths


def plot_saved_axis_orientation(numpy_folder, output_folder=None, **plot_options):
    """Public reader/plotter for full, reduced and legacy NumPy directories."""
    folder = Path(numpy_folder)
    output = Path(output_folder) if output_folder is not None else folder.parent / "Figures_axis_replot"
    return plot_axis_orientation_arrays(load_axis_arrays(folder), output, **plot_options)

def plot_axis_histogram(histogram, summary, path_png, dpi=200):
    """Signed fiber angles; each finite selected pixel contributes one count."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for index, axis in enumerate(("x", "y")):
        counts = histogram[:, 2 + index]
        fraction = counts / counts.sum() if counts.sum() else counts
        axes[index].bar(histogram[:, 0], fraction, width=histogram[:, 1]-histogram[:, 0],
                        align="edge", color="steelblue", edgecolor="white")
        axes[index].set(xlim=(-90, 90), xlabel=f"Angle towards {axis}-axis (deg)",
                        ylabel="Fraction of valid pixels",
                        title=f"Mean alignment: {summary[f'Alignment {axis}']:.3f}")
    fig.suptitle(f"Fiber orientation | {summary['Valid pixels']} valid pixels")
    fig.tight_layout()
    fig.savefig(path_png, dpi=dpi)
    plt.close(fig)


def plot_axis_channels(arrays, mask_image, mask, output_folder, scale=None, dpi=200):
    """Second channel and fiber channel side by side, with the reader's 25% overlay."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for axis in ("x", "y"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        angles = np.where(mask, arrays[f"min_{axis}"], np.nan)
        for ax, background, title in zip(axes, (mask_image, arrays["fiber"]),
                                         ("Mask channel", "Fiber channel")):
            ax.imshow(background, cmap="gray")
            ax.imshow(angles, cmap="gist_rainbow", vmin=-90, vmax=90, alpha=0.25)
            ax.set_title(title)
            ax.set_axis_off()
            if scale is not None:
                ax.add_artist(ScaleBar(scale, "um", length_fraction=0.2, location="lower right",
                                       color="white", box_alpha=0))
        # Opaque colorbar: alpha belongs to the image overlay, not the legend.
        mappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(-90, 90), cmap="gist_rainbow")
        fig.colorbar(mappable, ax=axes, orientation="horizontal", fraction=0.05, pad=0.03,
                     label=f"Fiber angle towards {axis}-axis (deg)")
        fig.savefig(output_folder / f"Channels_angles_towards_{axis}_axis.png", dpi=dpi)
        plt.close(fig)
    fig, ax = plt.subplots()
    ax.imshow(mask_image, cmap="gray")
    ax.imshow(np.ma.masked_where(~mask, mask), cmap="Greens", vmin=0, vmax=1, alpha=0.25)
    ax.set_title("Selected regions (green); all other pixels excluded")
    ax.set_axis_off()
    fig.savefig(output_folder / "ValidityMask.png", dpi=dpi)
    plt.close(fig)


def scale_for_quiver(ar1, ar2, dims, scale_ratio=0.2, return_scale=False):
    scale = scale_ratio * np.max(dims) / np.nanmax(np.sqrt((ar1) ** 2 + (ar2) ** 2))
    if return_scale:
        return scale
    return ar1 * scale, ar2 * scale


def add_colorbar(vmin, vmax, cmap="rainbow", ax=None, cbar_style="not-clickpoints", cbar_width="2%",
                 cbar_height="50%", cbar_borderpad=0.1, cbar_tick_label_size=15, cbar_str="",
                 cbar_axes_fraction=0.2, shrink=0.8, aspect=20, cbar_title_pad=1, **kwargs):
    from contextlib import suppress
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colormaps.get_cmap(cmap), norm=norm)
    sm.set_array([])  # bug fix for lower matplotlib version
    if cbar_style == "clickpoints":  # colorbar inside of the plot
        cbaxes = inset_axes(ax, width=cbar_width, height=cbar_height, loc=5, borderpad=cbar_borderpad * 30)
        cb0 = plt.colorbar(sm, cax=cbaxes)
        with suppress(TypeError, AttributeError):
            cbaxes.set_title(cbar_str, color="white", pad=cbar_title_pad)
        cbaxes.tick_params(colors="white", labelsize=cbar_tick_label_size)
    else:  # colorbar outide of the plot
        cbaxes = inset_axes(ax, width=cbar_width, height=cbar_height, loc=5, borderpad=cbar_borderpad * 30)
        cb0 = plt.colorbar(sm, aspect=aspect, shrink=shrink, fraction=cbar_axes_fraction,
                           pad=cbar_borderpad,cax=cbaxes)  # just exploiting the axis generation by a plt.colorbar
        cb0.outline.set_visible(False)
        cb0.ax.tick_params(labelsize=cbar_tick_label_size)
        with suppress(TypeError, AttributeError):
            cb0.ax.set_title(cbar_str, color="black", pad=cbar_title_pad)
    return cb0

def copycmap(cname):
    cmap =  plt.get_cmap(cname)
    colors = cmap(np.linspace(0, 1, 256))
    cmap =  matplotlib.colors.LinearSegmentedColormap.from_list(cname, cmap(np.linspace(0, 1, 256)))
    return cmap

def set_vmin_vmax(x, vmin, vmax):
    if not isinstance(vmin, (float, int)):
        vmin = np.nanmin(x)
    if not isinstance(vmax, (float, int)):
        vmax = np.nanmax(x)
    if isinstance(vmax, (float, int)) and not isinstance(vmin, (float, int)):
        vmin = vmax - 1 if vmin > vmax else None
    return vmin, vmax


def filter_values(ar1, ar2, abs_filter=0, f_dist=3):
    '''
    function to filter out values from an array for better display
    :param ar1:
    :param ar2:
    :param ar:
    :param f_dist: distance betweeen filtered values
    :return:
    '''
    # ar1_show=np.zeros(ar1.shape)+np.nan
    # ar2_show=np.zeros(ar2.shape)+np.nan
    pixx = np.arange(np.shape(ar1)[0])
    pixy = np.arange(np.shape(ar1)[1])
    xv, yv = np.meshgrid(pixy, pixx)

    def_abs = np.sqrt((ar1 ** 2 + ar2 ** 2))
    select_x = ((xv - 1) % f_dist) == 0
    select_y = ((yv - 1) % f_dist) == 0
    
     # Ignore RuntimeWarnings here 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        select_size = def_abs > abs_filter
    select = select_x * select_y * select_size
    s1 = ar1[select]
    s2 = ar2[select]
    return s1, s2, xv[select], yv[select]


def show_quiver(fx, fy, filter=[0, 1], scale_ratio=0.5, headwidth=0., headlength=0., headaxislength=0.,
                width=None, cmap="rainbow",
                figsize=None, cbar_str="", ax=None, fig=None
                , vmin=None, vmax=None, cbar_axes_fraction=0.2, cbar_tick_label_size=15
                , cbar_width="2%", cbar_height="25%", cbar_borderpad=0.1,
                cbar_style="not-clickpoints", plot_style="not-clickpoints", cbar_title_pad=1, plot_cbar=True, alpha=1,
                ax_origin="upper", pivot = "middle", **kwargs):
    # list of all necessary quiver parameters
    quiver_parameters = {"headwidth": headwidth, "headlength": headlength, "headaxislength": headaxislength,
                         "width": width, "scale_units": "xy", "angles": "xy", "scale": None}
    quiver_parameters = {key: value for key, value in quiver_parameters.items() if not value is None}

    fx = fx.astype("float64")
    fy = fy.astype("float64")
    dims = fx.shape  # needed for scaling
    if not isinstance(ax, matplotlib.axes.Axes):
        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
    map_values = np.sqrt(fx ** 2 + fy ** 2)
    vmin, vmax = set_vmin_vmax(map_values, vmin, vmax)
    im = plt.imshow(map_values, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, origin=ax_origin)  # imshowing
    if plot_style == "clickpoints":
        ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    # plotting arrows
    fx, fy, xs, ys = filter_values(fx, fy, abs_filter=filter[0],
                                   f_dist=filter[1])  # filtering every n-th value and every value smaller then x
    if scale_ratio:  # optional custom scaling with the image axis lenght
        fx, fy = scale_for_quiver(fx, fy, dims=dims, scale_ratio=scale_ratio)
        quiver_parameters["scale"] = 1  # disabeling the auto scaling behavior of quiver
    plt.quiver(xs, ys, fx, fy, **quiver_parameters)  # plotting the arrows
    if plot_cbar:
        add_colorbar(vmin, vmax, cmap, ax=ax, cbar_style=cbar_style, cbar_width=cbar_width, cbar_height=cbar_height,
                     cbar_borderpad=cbar_borderpad, v=cbar_tick_label_size, cbar_str=cbar_str,
                     cbar_axes_fraction=cbar_axes_fraction, cbar_title_pad=cbar_title_pad)
    return fig, ax


      
def plot_angle_dev(angle_map,vec0,vec1,coherency_map,path_png,label="Angle Deviation",dpi=300,cmap="viridis",vmin=None,vmax=None):
     # angle deviation no weights
    fig =plt.figure();
    if (vmin and vmax) is not None:
        plt.imshow(angle_map,vmin=vmin,vmax=vmax,cmap=cmap)
    else:
        plt.imshow(angle_map,cmap=cmap)
    cbar =plt.colorbar()
    mx = vec0 * coherency_map
    my = vec1 * coherency_map
    mx, my, x, y = filter_values(mx, my, abs_filter=0,
                                   f_dist=15)  #9
    plt.quiver(x, y, mx*300, my*300,scale=1.2,scale_units="xy", angles="xy",
               headwidth=0., headlength=0., headaxislength=0.)  #, width=0.005)
    plt.tight_layout()
    plt.axis('off'); cbar.set_label(label,fontsize=12)
    plt.tight_layout()
    plt.savefig(path_png, dpi=dpi, bbox_inches='tight', pad_inches=0);
    return fig
    
def plot_coherency(coherency,path_png,label="Coherency",dpi=300,vmin=None,vmax=None,cmap="viridis"):
     # angle deviation no weights
    fig =plt.figure()
    if (vmin and vmax) is not None:
        plt.imshow(coherency,vmin=vmin,vmax=vmax,cmap=cmap)
    else:
        plt.imshow(coherency,cmap=cmap)
    cbar =plt.colorbar()
    plt.tight_layout()
    plt.axis('off'); cbar.set_label(label,fontsize=12)
    plt.savefig(path_png, dpi=dpi, bbox_inches='tight', pad_inches=0); plt.tight_layout()
    return fig


def plot_fancy_overlay(fiber, cell, field,path_png,label="field",dpi=300,cmap_cell="Greys_r",cmap_fiber="Greys_r",
                       cmap_angle="viridis", alpha_ori =0.8,  alpha_cell = 0.4, alpha_fiber = 0.4,   # example cmaps: viridis/inferno/coolwarm 
                       omin=-1, omax=1, scale=None):                                
    fig =plt.figure()
    plt.imshow(field,cmap=cmap_angle,vmin=omin,vmax=omax,alpha=alpha_ori);cbar =plt.colorbar()
    plt.imshow(fiber, cmap=cmap_fiber,alpha=alpha_fiber) 
    if cell is not None:
        plt.imshow(cell, cmap=cmap_cell,alpha=alpha_cell) 
    plt.axis('off');
    cbar.set_label(label,fontsize=12)
    if scale is not None:
        scalebar = ScaleBar(scale, "um", length_fraction=0.1, location="lower right", box_alpha=0 , 
                    color="k")
        plt.gca().add_artist(scalebar)    
    plt.tight_layout()   
    plt.savefig(path_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
    return fig


     
def plot_polar(angle_plotting, something, path_png,label="something",dpi=300,vmin=None,vmax=None,
               something2 = None, something3 = None, label2 = None, label3 =None):
    fig = plt.figure();ax1 = plt.subplot(111, projection="polar")
    ax1.plot(angle_plotting, something, label=label , linewidth=2, c = "C0")
    if something2:
        ax1.plot(angle_plotting, something2, label=label2 , linewidth=2, c = "C1")
    if something3:
        ax1.plot(angle_plotting, something3, label=label3 , linewidth=2, c = "C2")    
    if (vmin and vmax) is not None:
        plt.ylim(vmin,vmax)
    plt.tight_layout();plt.legend(fontsize=7,frameon=False);plt.savefig(path_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
    return fig    

def plot_triple(results_angle, results_total, path_png ,dpi=300):
           # Triple plot
           fig = plt.figure(figsize=(20,6))
           axs1 = plt.subplot(131, projection="polar")
           axs1.plot(results_angle['Angles Plotting'],  results_angle['Coherency'], label="Coherency" )
           axs1.plot(results_angle['Angles Plotting'],  results_angle['Coherency (weighted by intensity)'], label="Coherency (weighted)" )
           plt.legend(fontsize=8,frameon=False); plt.ylim(0,1)
           ax2 = plt.subplot(132, projection="polar")
           ax2.plot(results_angle['Angles Plotting'], results_angle['Orientation'] , label="Orientation")
           ax2.plot(results_angle['Angles Plotting'], results_angle['Orientation (weighted by intensity and coherency)'] , label="Orientation weighed") 
           plt.legend(fontsize=8,frameon=False); plt.ylim(-1,1)
           strings = ["Mean Coherency", "Mean Coherency\nweighted by intensity", "Mean Angle Dev.",
                      "Mean Angle Dev. weighted\nby coherency", "Mean Angle Dev. weighted\nby coherency and intensity",
                      "Mean Orientation",
                      "Mean Orientation weighted\nby coherency",
                      "Mean Orientation weighted\nby coherency and intensity"]
           values = [results_total['Mean Coherency'][0] , results_total['Mean Coherency (weighted by intensity)'][0], 
                     results_total['Mean Angle'][0],
                     results_total['Mean Angle (weighted by coherency)'][0],
                     results_total['Mean Angle (weighted by intensity and coherency)'][0],
                     results_total['Orientation'][0],
                     results_total['Orientation (weighted by coherency)'][0],
                     results_total['Orientation (weighted by intensity and coherency)'][0],  
                     ]
           values = [[str(np.round(x,4))] for x in values]
           table_text = [[strings[i], values[i]] for i in range(len(values))]
           ax3 = plt.subplot(133);ax3.axis('tight'); ax3.axis('off')
           table = ax3.table(cellText = values, rowLabels=strings,bbox=[0.6,0.2,0.7,0.9])
           table.set_fontsize(11)
           #plt.tight_layout()
           plt.savefig(path_png, dpi=dpi)#, bbox_inches='tight', pad_inches=0)
           return fig

def quiv_coherency_center(vec0,vec1,center0,center1,coherency_map, path_png, dpi=200):
    f = np.nanpercentile(coherency_map,0.75)
    fig5, ax5 = show_quiver (vec0 * coherency_map, vec1 * coherency_map, filter=[f, 15], scale_ratio=0.1,width=0.003, 
                             cbar_str="Coherency", cmap="viridis")
    ax5.plot(center0,center1,"o")
    plt.tight_layout()
    plt.savefig(path_png, dpi=dpi, pad_inches=0)
    return fig5

def plot_fiber_seg(fiber_image,c0,c1,segmention, path_png,dpi=200, scale=None ):
    fig7= plt.figure() 
    my_norm = matplotlib.colors.Normalize(vmin=0.9999, vmax=1, clip=False)  
    # create a copy of matplotlib cmap
    cmap = copycmap("Greys")


    # everything under vmin gets transparent (all zeros in mask)
    cmap.set_under('k', alpha=0)
    #everything else visible
    cmap.set_over('k', alpha=1)
    # plot mask and center
    plt.imshow(fiber_image, origin="upper")
    if segmention is not None:
        plt.imshow(segmention, cmap=cmap, norm = my_norm, origin="upper")
    plt.scatter(c0,c1, c= "w");plt.axis('off');
    if scale is not None:
        scalebar = ScaleBar(scale, "um", length_fraction=0.1, location="lower right", box_alpha=0 , 
                    color="k")
        plt.gca().add_artist(scalebar)

    plt.tight_layout() ;plt.savefig(path_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
    return fig7

def plot_overlay(fiber_image , c0,c1, vec0,vec1, coherency_map,
                       segmention,path_png,dpi=200, show_n = 15, scale=None ):
            fig=plt.figure();f = np.nanpercentile(coherency_map,0.75)
            # plot overlay
            my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
            # create a copy of matplotlib cmap
            cmap = copycmap("Greys")
            # everything under vmin gets transparent (all zeros in mask)
            cmap.set_under('k', alpha=0)
            #everything else visible
            cmap.set_over('k', alpha=1)
            # plot mask and center
            show_quiver (vec0 * coherency_map, vec1 * coherency_map, filter=[f, show_n],alpha=0 , scale_ratio=0.08,width=0.002, plot_cbar=False, cbar_str="coherency", cmap="viridis")
            plt.imshow(fiber_image, origin="upper")
            if segmention is not None:
                plt.imshow(segmention, cmap=cmap, norm = my_norm, origin="upper", zorder=100)
                plt.scatter(c0,c1, c= "w", zorder=200)
            if scale is not None:
                scalebar = ScaleBar(scale, "um", length_fraction=0.1, location="lower right", box_alpha=0 , 
                            color="k")
                plt.gca().add_artist(scalebar)
            plt.tight_layout() ;plt.savefig(path_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
            return fig       
        
def plot_shells(shell_masks,path_png,dpi=200 ):
    fig =  plt.figure()
    cmap_list = ["Greens","Greys","Reds","Oranges","Blues","PuBu","GnBu"]
    for s in  range(len(shell_masks)):
        my_norm = matplotlib.colors.Normalize(vmin=0.99, vmax=1, clip=False)  
        # create a copy of matplotlib cmap
        cmap = copycmap(cmap_list[s%len(cmap_list)])
        # everything under vmin gets transparent (all zeros in mask)
        cmap.set_under('k', alpha=1)
        #everything else visible
        cmap.set_over('k', alpha=0)
        # plot mask and center
        plt.imshow(shell_masks[-s], cmap = cmap ,  origin="upper", alpha= 0.2 ) 
    plt.tight_layout(); plt.axis("off");plt.savefig(path_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
    return fig
    
 # Distance analysis
def plot_distance(results_distance,path_png,string_plot = "Orientation (individual)", ylabel=None, dpi=200, ylim=None):
    if not ylabel:
         ylabel= string_plot
    fig = plt.figure()
    plt.plot(results_distance['Shell_mid (µm)'],results_distance[string_plot],"o-", c="lightgreen",label="Orientation")
    plt.grid(); plt.xlabel("Distance (µm)"); plt.ylabel(ylabel)
    plt.tight_layout();plt.savefig(path_png, dpi=dpi)
    if ylim:
        plt.ylim(ylim)
    return fig
           
def plot_cell(im_cell_n,  path_png,scale=None,dpi=200):
        fig = plt.figure(); plt.imshow(im_cell_n, cmap="viridis");plt.axis("off"),plt.tight_layout()
        if scale is not None:
            scalebar = ScaleBar(scale, "um", length_fraction=0.1, location="lower right", box_alpha=0 , 
                        color="k")
        plt.gca().add_artist(scalebar)
        plt.savefig(path_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
        return fig
