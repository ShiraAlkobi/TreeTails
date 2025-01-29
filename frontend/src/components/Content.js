import React, { useState } from "react";
import "../styles/Home.css";
import OutputBox from './OutputBox';
import UploadSection from './ImageUploader';

const Content = () => {
    const [image, setImage] = useState(null);  // Stores image file
    const [response, setResponse] = useState("");  // Stores backend response

    const sendForAnalysis = async () => {
        if (!image) {
            alert("Please upload an image first.");
            return;
        }

        const formData = new FormData();
        formData.append("file", image); // Attach the image file

        try {
            const res = await fetch("http://127.0.0.1:5000/upload", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                throw new Error(`Server error: ${res.status}`);
            }

            const data = await res.json();
            setResponse(data.personality_traits); // Store personality traits in state
        } catch (error) {
            console.error("Error analyzing image:", error);
            setResponse("An error occurred. Please try again.");
        }
    };

    return (
        <div className="content">
            <UploadSection setImage={setImage} sendForAnalysis={sendForAnalysis} />
            <OutputBox response={response} />
        </div>
    );
};

export default Content;
