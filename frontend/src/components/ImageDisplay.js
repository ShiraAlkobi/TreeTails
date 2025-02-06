import React from "react";
import "../styles/FriendsPage.css";

const ImageDisplay = ({ image, response }) => {
    return (
        <div className="image-display-container">
            {image && <img src={URL.createObjectURL(image)} alt="Uploaded" className="analyzed-image" />}  {/* Display the saved image */}
            {response && (
                <div className="analysis-summary">
                    <h3>סיכום הניתוח:</h3>
                    <p>{response}</p>  {/* Display the saved response */}
                </div>
            )}
        </div>
    );
};

export default ImageDisplay;
