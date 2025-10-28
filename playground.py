import xarray as xr
import matplotlib.pyplot as plt




def plotting(array):

    # --- Assume 'orog_values' variable already exists ---
    # Example: If you need to re-create it for testing:
    # orog_values = np.random.rand(1069, 1069) * 330 - 5
    # ----------------------------------------------------

    print("Plotting array with corrected orientation...")

    # Use imshow to create an image plot from the 2D array
    # Adding 'origin="lower"' will display the (0,0) index at the bottom-left
    # plt.imshow(array, cmap='terrain', origin='lower')
    plt.imshow(array, cmap='terrain')

    # Add a color bar to show the scale of the values
    plt.colorbar(label='Orography Value')

    # Add a title and labels (optional, but good practice)
    plt.title('Orography Data (Corrected Orientation)')
    plt.xlabel('X Index')
    plt.ylabel('Y Index')

    # Save the plot to a file
    output_filename = 'orog_plot_corrected.png'
    plt.savefig(output_filename)

    # Close the plot to free up memory
    plt.close()

    print(f"Plot successfully saved to {output_filename}")
    

if __name__ == "__main__":
    # Load the dataset
    ds = xr.open_dataset("ERA5_download/allEurope/single_levels/2013.grib")

    # Extract the 'orog' variable
    orog_values = ds['z'].values #(1, H, W) -> (H, W)
    # orog_values = orog_values.squeeze()  # Remove singleton dimensions if any

    # Call the plotting function with the extracted data
    plotting(orog_values)
    
    
    