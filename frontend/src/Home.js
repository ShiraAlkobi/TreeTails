import React, { useState } from 'react';
import './Home.css';

const Home = () => {
    const [image, setImage] = useState(null);

    const handleImageUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            const imageUrl = URL.createObjectURL(file);
            setImage(imageUrl);
        }
    };

    return (
        <div className="container">
            <h1>🌿 Welcome to TreeTails! 🌈</h1>
            <div className="upload-section">
                <label className="upload-btn">
                    Upload an Image 📸
                    <input type="file" accept="image/*" onChange={handleImageUpload} />
                </label>
            </div>
            <div className="output-section">
                {image ? <img src={image} alt="Uploaded Preview" className="preview" /> : <p>No image uploaded yet.</p>}
            </div>
        </div>
    );
};

export default Home;
