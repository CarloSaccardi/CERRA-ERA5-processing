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

    # Add a color bar to show the scale of the values m2 s-2
    plt.colorbar(label='m2 s-2')

    # Add a title and labels (optional, but good practice)
    plt.title('Geopotential')
    
    plt.axis('off')

    # Save the plot to a file
    output_filename = 'ERA5_Iberia.png'
    plt.savefig(output_filename)

    # Close the plot to free up memory
    plt.close()

    print(f"Plot successfully saved to {output_filename}")
    

if __name__ == "__main__":
    # Load the dataset
    ds = xr.open_dataset("zz_processed_data/ERA5/test/static_Iberia.nc")

    # Extract the 'orog' variable
    # Select the 'z' variable, then select the 0th index from the 'valid_time' dimension
    orog_values = ds["geopotential"].values
    # orog_values = orog_values.squeeze()  # Remove singleton dimensions if any

    # Call the plotting function with the extracted data
    plotting(orog_values)
    
    
    