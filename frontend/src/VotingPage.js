import React, { useState } from "react";
import { useAppContext } from "./AppContext";
import "./styles/VotingPage.css"; // Ensure the CSS is imported

const VotingPage = () => {
    const { response } = useAppContext(); // Get the response from the context

    // Set the initial ratings (an array of zeros for now)
    const [ratings, setRatings] = useState(
        new Array(response.split("\n").length).fill(0)
    );

    // This will hold all the votes, initialized as an empty array
    const [allVotes, setAllVotes] = useState([]);

    // Handle star clicks
    const handleRating = (index, rating) => {
        const newRatings = [...ratings];
        newRatings[index] = rating;
        setRatings(newRatings);
    };

    // Save the ratings to a file (appending to the existing list of votes)
    const saveRatings = () => {
        // Add the current ratings to the array of all votes
        setAllVotes((prevVotes) => {
            const updatedVotes = [...prevVotes, [...ratings]];
            return updatedVotes;
        });

        // Prepare the data for download, containing all the votes
        const data = JSON.stringify(allVotes);
        const blob = new Blob([data], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "all_ratings.json"; // Name of the file
        link.click();
    };

    if (!response) {
        return <div>לא ניתן לטעון את המידע, נסה שוב מאוחר יותר.</div>; // Show error message if response is not available
    }

    return (
        <div className="voting-page-container">
            <h1>הצבע כמה המודל שלנו צדק</h1>

            {/* Voting Table */}
            <table className="voting-table">
                <thead>
                    <tr>
                        <th>נקודה מתוך הניתוח</th>
                        <th>הצבעה</th>
                    </tr>
                </thead>
                <tbody>
                    {response.split("\n\n").map((line, index) => (
                        <tr key={index}>
                            <td>{line}</td>
                            <td className="star-column">
                                {[1, 2, 3, 4, 5].map((star) => (
                                    <span
                                        key={star}
                                        className="star"
                                        onClick={() => handleRating(index, star)}
                                        style={{
                                            color: ratings[index] >= star ? "gold" : "gray",
                                        }}
                                    >
                                        ★
                                    </span>
                                ))}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>

            <button className="save-button" onClick={saveRatings}>שמור את הצבעותיך</button> {/* Save button */}
        </div>
    );
};

export default VotingPage;
