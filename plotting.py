import matplotlib.pyplot as plt

def generate_astro_plot(data, wcs):
    """ Generate an astro image plot using matplotlib using the correct
    orientation for astro images (pixel coord origin in the lower left)
    left handed world coord system with longitude increasing to the left (east)
    and latitude increasing upwards (north).
    input the pixel data and the associated WCS object to correctly orient the plot
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=wcs)

    # IMPORTANT: data.T is required because imshow expects [row, col] (y, x)
    # but our data is [x, y].
    ax.imshow(data.T, origin='lower', cmap='magma')
    return fig, ax

    """
    # Draw Arrow showing the major axis in pixel space
    x0, y0 = 100, 100
    length = 40
    ax.arrow(
        x0, y0, length*np.cos(np.radians(recovered_theta)),
        length*np.sin(np.radians(recovered_theta)),
        color='cyan', width=1, head_width=5, label='Major Axis'
    )

    ax.coords[0].set_axislabel('l')
    ax.coords[0].set_axislabel('m')
    ax.coords[1].set_axislabel('Declination')
    plt.title(f"Sky PA: {pa:.2f}° | Pixel Theta: {recovered_theta:.2f}°")
    plt.legend()
    return plt
    """

