import React, { useRef, useEffect, useState } from "react";
import "../styles/Home.css";
import OutputBox from "./OutputBox";
import UploadSection from "./ImageUploader";
import DownloadIcon from "../images/download.png";
import SendIcon from "../images/send.png";
import { useAppContext } from "../AppContext"; // Import the custom hook for context

const Content = () => {
    const { image, setImage, response, setResponse } = useAppContext(); // Access image and response from context
    const [isPopupVisible, setIsPopupVisible] = useState(false); // Controls popup visibility
    const canvasRef = useRef(null);

    const sendForAnalysis = async () => {
        if (!image) {
            alert("Please upload an image first.");
            return;
        }

        const formData = new FormData();
        formData.append("file", image);

        try {
            const res = await fetch("http://127.0.0.1:5000/upload", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                throw new Error(`Server error: ${res.status}`);
            }

            const data = await res.json();
            setResponse(data.personality_traits); // Set the response in context

        } catch (error) {
            console.error("Error analyzing image:", error);
            setResponse("An error occurred. Please try again.");
        }
    };

    const handleShareClick = () => {
        setIsPopupVisible(true); // Show the popup when the share button is clicked
    };

    const closePopup = () => {
        setIsPopupVisible(false); // Close the popup when the close button is clicked
    };

    const createImagePreview = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        if (image && response) {
            const img = new Image();
            img.src = URL.createObjectURL(image);
            img.onload = () => {
                // Clear the canvas before drawing
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Draw the background with a gradient (nature-inspired)
                const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
                gradient.addColorStop(0, "#A5D6A7"); // Light green
                gradient.addColorStop(1, "#81C784"); // Darker green
                ctx.fillStyle = gradient;
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                // Image width is now 40% of canvas width, placed on the right
                const imageWidth = canvas.width * 0.4; // 40% of the canvas width
                const imageHeight = canvas.height;  // Full height of the canvas

                // Draw the image on the right side (40% width, right-aligned)
                ctx.drawImage(img, canvas.width - imageWidth, 0, imageWidth, imageHeight);

                // Set the font size and style for the text
                ctx.fillStyle = "white";
                ctx.font = "10px 'Noto Sans Hebrew', serif";  // Adjust font size if necessary

                // Set the direction for RTL (Right to Left) for Hebrew text
                ctx.direction = "rtl";  // RTL direction for Hebrew text
                ctx.textBaseline = "top"; // Text starting from the top

                // Add a header for the image on the left side (60% of the canvas)
                ctx.fillStyle = "#388E3C"; // Darker green for header
                ctx.font = "17px 'Noto Sans Hebrew', serif";  // Larger font for the header
                ctx.fillText("הניתוח הייחודי שלי מ - TreeTails", canvas.width * 0.6 - 20, 20); // Place header on the left

                // Function to wrap text for RTL and ensure it fits within the given width
                const maxWidth = canvas.width * 0.62 - 30;  // Limit text width to 60% of canvas width
                const lineHeight = 30;  // Line height for text

                // Split the response text into words
                const words = response.split(" ");
                let line = "";
                let lines = [];

                // Split the response text into lines that fit within the maxWidth
                for (let i = 0; i < words.length; i++) {
                    const testLine = line + words[i] + " ";
                    const metrics = ctx.measureText(testLine);
                    if (metrics.width > maxWidth) {
                        lines.push(line);
                        line = words[i] + " ";
                    } else {
                        line = testLine;
                    }
                }

                lines.push(line);  // Add the last line

                // Draw each line of text starting from below the header
                let y = 70;  // Starting position for the text (below the header)
                for (let i = 0; i < lines.length; i++) {
                    ctx.fillStyle = "#388E3C";  // A darker green for body text
                    ctx.fillText(lines[i], canvas.width * 0.6 - 20, y);  // Place text on the left side
                    y += lineHeight;
                }
            };
        }
    };

    // Trigger the image preview generation when the image or response changes
    useEffect(() => {
        if (isPopupVisible) {
            createImagePreview();
        }
    }, [image, response, isPopupVisible]);

    // Share functionality (for WhatsApp and download)
    const shareOnWhatsApp = () => {
        const imageUrl = canvasRef.current.toDataURL(); // Get canvas image URL
        const encodedUrl = encodeURIComponent(imageUrl);
        window.open(`https://wa.me/?text=${encodedUrl}`, "_blank");
    };

    const downloadImage = () => {
        const imageUrl = canvasRef.current.toDataURL(); // Get canvas image URL
        const link = document.createElement("a");
        link.href = imageUrl;
        link.download = "analysis-image.png";
        link.click();
    };

    return (
        <div className="content">
            <UploadSection setImage={setImage} sendForAnalysis={sendForAnalysis} />
            <OutputBox response={response} />

            {image && response && (
                <button className="share-button" onClick={handleShareClick}>
                    ?רוצה לשתף
                </button>
            )}

            {/* Popup for Image Preview */}
            {isPopupVisible && (
                <div className="popup-container">
                    <div className="popup-content">
                        <canvas
                            ref={canvasRef}
                            width={600}  // Adjust size for your template
                            height={400}
                            className="image-preview"
                        ></canvas>

                        <div className="popup-buttons">
                            <button className="whatsapp-button" onClick={shareOnWhatsApp}>
                               <img className="popup-icons" src={SendIcon}/>
                            </button>
                            <button className="download-button" onClick={downloadImage}>
                                <img className="popup-icons" src={DownloadIcon}/>
                            </button>
                        </div>

                        <button className="close-popup" onClick={() => setIsPopupVisible(false)}>
                            X
                        </button>
                    </div>
                    <div className="popup-overlay" onClick={() => setIsPopupVisible(false)}></div>
                </div>
            )}
        </div>
    );
};

export default Content;
