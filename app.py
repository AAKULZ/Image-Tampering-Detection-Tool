import os
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file, session, redirect, url_for
from PIL import Image, ImageChops, ImageEnhance, ImageOps, ExifTags
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Upload and static folder configuration
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Extract metadata from an image file
def get_exif_data(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    image_metadata = {}
    if exif_data:
        for tag, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            image_metadata[tag_name] = str(value) if not isinstance(value, (int, float, str)) else value
    return image_metadata

# Analyze images in directory for metadata
def analyze_images_in_directory(directory_path):
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(tuple(ALLOWED_EXTENSIONS))]
    metadata_tables = {}
    edited_images = []
    non_edited_images = []
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        metadata = get_exif_data(image_path)
        edited = "No"
        if not metadata or "DateTime" not in metadata or "Model" not in metadata:
            edited = "Yes"
            edited_images.append(image_file)
        else:
            non_edited_images.append(image_file)
        metadata['File Name'] = image_file
        metadata['Edited'] = edited
        metadata_df = pd.DataFrame(list(metadata.items()), columns=['Property', 'Value'])
        metadata_tables[image_file] = metadata_df
    return metadata_tables, edited_images, non_edited_images

# Perform ELA analysis and save results
def perform_ela(image_path, compression_quality=90):
    original = Image.open(image_path).convert('RGB')
    compressed_path = os.path.join(app.config['STATIC_FOLDER'], 'compressed_image.jpg')
    original.save(compressed_path, 'JPEG', quality=compression_quality)
    compressed = Image.open(compressed_path).convert('RGB')

    # Calculate Error Level Analysis (ELA) image
    ela = ImageChops.difference(original, compressed)
    enhancer = ImageEnhance.Contrast(ela)
    ela = enhancer.enhance(10)
    ela_gray = ImageOps.grayscale(ela)
    np_ela = np.array(ela_gray)
    mean_ela, std_ela, max_ela = np.mean(np_ela), np.std(np_ela), np.max(np_ela)

    # Save ELA image
    ela_image_path = os.path.join(app.config['STATIC_FOLDER'], f'ela_{os.path.basename(image_path)}')
    ela.save(ela_image_path)

    # Save Histogram
    hist_path = os.path.join(app.config['STATIC_FOLDER'], f'hist_{os.path.basename(image_path)}')
    plt.figure(figsize=(6, 4))
    plt.hist(np_ela.ravel(), bins=50, color='blue', edgecolor='black')
    plt.xlabel('Error Level')
    plt.ylabel('Frequency')
    plt.title('ELA Value Histogram')
    plt.savefig(hist_path)
    plt.close()

    # Save Heatmap
    heatmap_path = os.path.join(app.config['STATIC_FOLDER'], f'heatmap_{os.path.basename(image_path)}')
    plt.figure(figsize=(6, 4))
    sns.heatmap(np_ela, cmap='hot', cbar=True)
    plt.title('ELA Heatmap')
    plt.savefig(heatmap_path)
    plt.close()

    # Clean up compressed image
    os.remove(compressed_path)

    return mean_ela, std_ela, max_ela, ela_image_path, hist_path, heatmap_path


# Flask Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('images')
    if not files:
        return "No files selected", 400
    
    for file in files:
        if file and allowed_file(file.filename):
            # Save file to the 'uploads' directory (backend)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(upload_path)
            
            # Open the image file using PIL to ensure it is processed correctly
            img = Image.open(upload_path)

            # Save a copy of the image in the 'static/uploads' directory (for serving to clients)
            static_path = os.path.join(app.config['STATIC_FOLDER'], file.filename)

            # Ensure the image is saved in the correct format (JPEG, PNG, etc.)
            img.save(static_path, format=img.format)  # Ensures correct format (e.g., PNG, JPEG)

    return redirect(url_for('index'))



@app.route('/view_metadata')
def view_metadata():
    metadata_tables, edited_images, non_edited_images = analyze_images_in_directory(app.config['UPLOAD_FOLDER'])
    flattened_metadata = {}
    for image_file, metadata_df in metadata_tables.items():
        metadata_list = [{'Property': row['Property'], 'Value': row['Value']} for index, row in metadata_df.iterrows()]
        flattened_metadata[image_file] = metadata_list
    session['flattened_metadata'] = flattened_metadata
    session['edited_images'] = edited_images
    session['non_edited_images'] = non_edited_images
    return render_template('results.html', metadata_tables=metadata_tables, edited_images=edited_images, non_edited_images=non_edited_images)

def analyze_ela_in_directory(directory_path):
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(tuple(ALLOWED_EXTENSIONS))]
    ela_data = {}

    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        mean_ela, std_ela, max_ela, ela_image_path, hist_path, heatmap_path = perform_ela(image_path)

        # Store the file names without modifying the path
        ela_data[image_file] = {
            'Mean Error Level': mean_ela,
            'Standard Deviation': std_ela,
            'Maximum Error Level': max_ela,
            'ELA Image Path': f'uploads/{os.path.basename(ela_image_path)}',
            'Histogram Path': f'uploads/{os.path.basename(hist_path)}',
            'Heatmap Path': f'uploads/{os.path.basename(heatmap_path)}'
        }

    return ela_data

@app.route('/view_ela')
def view_ela():
    ela_data = analyze_ela_in_directory(app.config['UPLOAD_FOLDER'])
    
    ela_display_data = []
    for image_file, stats in ela_data.items():
        ela_display_data.append({
            'original': image_file,
            'mean': stats['Mean Error Level'],
            'std_dev': stats['Standard Deviation'],
            'max': stats['Maximum Error Level'],
            # Directly refer to the file paths in the static/uploads folder
            'ela_image': url_for('static', filename=stats['ELA Image Path']),
            'histogram': url_for('static', filename=stats['Histogram Path']),
            'heatmap': url_for('static', filename=stats['Heatmap Path'])
        })
    
    if not ela_display_data:
        return "No ELA results to display.", 404

    return render_template('ela_results.html', ela_display_data=ela_display_data)

@app.route('/download_report')
def download_report():
    flattened_metadata = session.get('flattened_metadata', {})
    metadata_list = [[image_file, item['Property'], item['Value']] for image_file, metadata in flattened_metadata.items() for item in metadata]
    df = pd.DataFrame(metadata_list, columns=['Image File', 'Property', 'Value'])
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, as_attachment=True, download_name="image_metadata_report.csv", mimetype="text/csv")

# Statistical Analysis function for SAR (Statistical Analysis Report)
def perform_statistical_analysis(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    colors = ('r', 'g', 'b')
    plot_paths = {}
    
    # Create histogram plots for each color channel
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
        plt.figure()
        plt.plot(hist, color=color)
        plt.title(f'{color.upper()} Channel Histogram')
        plt.xlim([0, 256])
        
        # Save each histogram as an image file
        hist_path = os.path.join(app.config['STATIC_FOLDER'], f'hist_{color}_{os.path.basename(image_path)}.png')
        plt.savefig(hist_path)
        plt.close()  # Close the figure to prevent displaying it
        plot_paths[f'{color}_hist'] = hist_path
    
    # Calculate color statistics
    mean_color = np.mean(image_rgb, axis=(0, 1))
    std_color = np.std(image_rgb, axis=(0, 1))
    median_color = np.median(image_rgb, axis=(0, 1))
    
    # Save bar plots for mean, standard deviation, and median
    stats_types = {'mean': mean_color, 'std_dev': std_color, 'median': median_color}
    for stat, values in stats_types.items():
        plt.figure()
        sns.barplot(x=['Red', 'Green', 'Blue'], y=values, palette='viridis')
        plt.title(f'{stat.capitalize()} Color Values')
        stat_path = os.path.join(app.config['STATIC_FOLDER'], f'{stat}_{os.path.basename(image_path)}.png')
        plt.savefig(stat_path)
        plt.close()
        plot_paths[f'{stat}_bar'] = stat_path

    return mean_color, std_color, median_color, plot_paths


def analyze_sla_in_directory(directory_path):
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(tuple(ALLOWED_EXTENSIONS))]
    stats_data = {}

    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        
        mean_color, std_color, median_color, plot_paths = perform_statistical_analysis(image_path)
        
        # Corrected paths by including 'uploads/' in the `filename`
        stats_data[image_file] = {
            'Mean Red': mean_color[0],
            'Mean Green': mean_color[1],
            'Mean Blue': mean_color[2],
            'Std Red': std_color[0],
            'Std Green': std_color[1],
            'Std Blue': std_color[2],
            'Median Red': median_color[0],
            'Median Green': median_color[1],
            'Median Blue': median_color[2],
            'histograms': {color: url_for('static', filename=f'uploads/{os.path.basename(path)}') for color, path in plot_paths.items()}
        }

    return stats_data



def format_stats_data(stats_data):
    stats_df = pd.DataFrame(stats_data).T
    stats_df.index.name = 'File Name'
    return stats_df

@app.route('/view_sar')
def view_sar():
    stats_data = analyze_sla_in_directory(app.config['UPLOAD_FOLDER'])

    # Convert stats_data into a format suitable for rendering in the template
    sar_display_data = []
    for image_file, stats in stats_data.items():
        sar_display_data.append({
            'original': image_file,
            'mean_red': stats['Mean Red'],
            'mean_green': stats['Mean Green'],
            'mean_blue': stats['Mean Blue'],
            'std_red': stats['Std Red'],
            'std_green': stats['Std Green'],
            'std_blue': stats['Std Blue'],
            'median_red': stats['Median Red'],
            'median_green': stats['Median Green'],
            'median_blue': stats['Median Blue'],
            'histograms': stats['histograms']
        })

    return render_template('view_sar.html', sar_display_data=sar_display_data)

# DCT-based forgery detection with saving of heatmaps and suspicious regions
def dct_based_forgery_detection(image_path, block_size=16, threshold=3):
    # Load the image and convert it to grayscale
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape

    # Initialize arrays to store DCT blocks and coefficients
    blocks = []
    dct_coefficients = []

    # Split the image into blocks and apply DCT
    for i in range(0, height - block_size, block_size):
        for j in range(0, width - block_size, block_size):
            block = gray_image[i:i+block_size, j:j+block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                dct_block = cv2.dct(np.float32(block))
                blocks.append(dct_block)
                dct_coefficients.append(dct_block.flatten())
    
    # Convert list of coefficients into a numpy array for statistical analysis
    dct_coefficients = np.array(dct_coefficients)

    # Perform Z-score analysis to detect anomalies
    z_scores = np.abs(zscore(dct_coefficients, axis=0))

    # Identify blocks with significant tampering
    suspicious_blocks = np.where(z_scores > threshold)

    # Save the DCT coefficient heatmaps for the first few blocks
    dct_heatmaps = []
    for idx, block in enumerate(blocks[:5]):
        heatmap_path = os.path.join(STATIC_FOLDER, f'dct_heatmap_{os.path.basename(image_path)}_block{idx+1}.png')
        plt.imshow(np.log(np.abs(block)), cmap='hot')
        plt.colorbar()
        plt.title(f"DCT Heatmap Block {idx+1}")
        plt.savefig(heatmap_path)
        plt.close()
        dct_heatmaps.append(heatmap_path)

    # Save visualization of suspicious regions on the image
    suspicious_image_path = os.path.join(STATIC_FOLDER, f'suspicious_{os.path.basename(image_path)}')
    suspicious_image = image.copy()
    for idx in suspicious_blocks[0]:
        i, j = divmod(idx, int(image.shape[1] / block_size))
        top_left = (j * block_size, i * block_size)
        bottom_right = ((j + 1) * block_size, (i + 1) * block_size)
        cv2.rectangle(suspicious_image, top_left, bottom_right, (0, 0, 255), 2)
    cv2.imwrite(suspicious_image_path, suspicious_image)

    return suspicious_image_path, dct_heatmaps

@app.route('/view_dct')
def view_dct():
    image_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
    dct_results = []

    for image_file in image_files:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file)
        
        # Perform DCT-based forgery detection
        suspicious_image_path, dct_heatmaps = dct_based_forgery_detection(image_path)
        
        # Store the DCT results for rendering
        dct_results.append({
            'original': image_file,
            'suspicious_image': url_for('static', filename=f'uploads/{os.path.basename(suspicious_image_path)}'),
            'dct_heatmaps': [url_for('static', filename=f'uploads/{os.path.basename(path)}') for path in dct_heatmaps]
        })

    return render_template('view_dct.html', dct_results=dct_results)




if __name__ == '__main__':
    app.run(debug=True)
