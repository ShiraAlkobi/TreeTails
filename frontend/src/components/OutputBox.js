import React from "react";
import "../styles/Home.css";

const OutputBox = ({ response }) => {
  return (
    <div className="output-box">
      {response ?
        <div className="output-box-text">
            {response}
        </div>
       : <div className="output-box-text">
        מודל הקסם שלנו מחכה לך, תוצאות הניתוח יופיעו כאן
        </div>}
    </div>
  );
};

export default OutputBox;