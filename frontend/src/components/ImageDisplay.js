import React from "react";
import "../styles/FriendsPage.css";

const ImageDisplay = () => {
    return (
        <div className="image-display-container">
            <img src="path/to/image.jpg" alt="Analyzed" className="analyzed-image" />
            <div className="analysis-summary">
                <h3>Analysis Summary</h3>
                <ul>
                    <li>Confidence: High</li>
                    <li>Stability: Medium</li>
                    <li>Creativity: Excellent</li>
                </ul>
            </div>
        </div>
    );
};

export default ImageDisplay;
