import { useState } from "react";
import binIcon from '../images/bin.png';
import treeUploadIcon from '../images/treeUploading.png';

const UploadSection = ({ setImage, sendForAnalysis }) => {
    const [imagePreview, setImagePreview] = useState(null); // Store preview image URL

    const handleImageUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            const imageUrl = URL.createObjectURL(file);
            setImagePreview(imageUrl);
            setImage(file); // Pass actual file to Content.js for backend processing
        }
    };

    const removeImage = () => {
        setImagePreview(null);
        setImage(null); // Clear the image in Content.js
    };

    return (
        <div className="upload-section">
            {/* Upload Button */}
            <label className="upload-button">
                העלה תמונה מהמחשב שלך
                <input type="file" accept="image/*" onChange={handleImageUpload} />
            </label>

            {/* Remove Image Button (Only appears when an image is uploaded) */}
            {imagePreview && (
                <button className="remove-icon-button" onClick={removeImage}>
                    <img src={binIcon} alt="Remove" className="remove-button-icon"/>
                </button>
            )}

            {/* Image Preview / Placeholder */}
            <div className="image-preview">
                {imagePreview ? (
                    <img src={imagePreview} alt="Uploaded" className="uploaded-image" />
                ) : (
                    <div className="placeholder-container">
                        <img src={treeUploadIcon} alt="Upload" className="upload-icon"/>
                        <p className="placeholder-text">!קדימה... בואו נתחיל</p>
                    </div>
                )}
            </div>

            {/* Send for Analysis Button */}
            <button className="analysis-button" onClick={sendForAnalysis}>
                תן למודל הקסם שלנו לגלות קצת עלייך
            </button>
        </div>
    );
};

export default UploadSection;
