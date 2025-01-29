import React from "react";
import "../styles/Home.css";
import drawingIcon from '../images/drawing.png';
import uploadingIcon from '../images/upload.png';
import aiIcon from '../images/AI.png';
import checkIcon from '../images/check.png';
import friendsIcon from '../images/friends.png';
import voteIcon from '../images/vote.png';

// Define the steps for using the app
const steps = [
  { icon: voteIcon, label: 'אימות', description: 'תן לחבריך להצביע ולשתף כמה צדקנו' },
  { icon: friendsIcon, label: 'שיתוף', description: 'שתף את הניתוח עם חבריך'},
  { icon: checkIcon, label: 'אישור', description: 'אשר את התוצאה ושמור את העבודה שלך' },
  { icon: aiIcon, label: 'חיזוי', description: 'תן למודל שלנו לגלות קצת עליך' },
  { icon: uploadingIcon, label: 'העלאה', description: 'העלה את העיצוב לאפליקציה שלנו' },
  { icon: drawingIcon, label: 'ציור', description: 'התחל בציור העץ היחודי שלך' }
];

const header = '?אז איך עושים את זה';

const UseSteps = () => {
  return (
    <div className="use-steps">
      <h2>{header}</h2>
      <div className="steps-container">
        {steps.map((step, index) => (
          <div key={index} className="step">
            <div className="step-label">{step.label}</div>
            <div className="step-icon">
              <img src={step.icon} alt={step.label} className="step-icon img" />
            </div>
            <div className='step-description'>
              <p>{step.description}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default UseSteps;
