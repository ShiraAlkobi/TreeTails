import React from "react";
import "../styles/FriendsPage.css";

const VotingTable = ({ analysisPoints, ratings }) => {

    // Function to generate star ratings
    const renderStars = (rating) => {
        const maxStars = 5;
        return (
            <span className="stars">
                {"★".repeat(rating)}{"☆".repeat(maxStars - rating)}
            </span>
        );
    };

    return (
        <div className="voting-table-container">
            <table>
                <thead>
                    <tr>
                        <th>?מה מודל הקסם שלנו קבע</th>
                        <th>?כמה צדקנו</th>
                    </tr>
                </thead>
                <tbody>
                    {analysisPoints.map((point, index) => (
                        <tr key={index}>
                            <td>{point}</td>
                            <td>{renderStars(ratings[index])}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default VotingTable;
