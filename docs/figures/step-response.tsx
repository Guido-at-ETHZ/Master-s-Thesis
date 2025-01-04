import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

const FirstOrderSystem = () => {
  const [tau, setTau] = useState(1);
  const [targetValue, setTargetValue] = useState(1);
  const [data, setData] = useState([]);
  const [stepStartTime, setStepStartTime] = useState(0);
  const [stepStartValue, setStepStartValue] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  
  useEffect(() => {
    // When target value changes, store current state as starting point
    setStepStartTime(currentTime);
    setStepStartValue(data.length > 0 ? data[data.length - 1].value : 0);
  }, [targetValue]);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(prevTime => {
        const newTime = prevTime + 0.1;
        const elapsedTime = newTime - stepStartTime;
        
        // First order response from current value to target
        const currentValue = stepStartValue + 
          (targetValue - stepStartValue) * (1 - Math.exp(-elapsedTime/tau));
        
        setData(prevData => {
          const newData = [...prevData, {
            time: newTime.toFixed(1),
            value: currentValue
          }];
          if (newData.length > 100) newData.shift();
          return newData;
        });
        
        return newTime;
      });
    }, 100);
    
    return () => clearInterval(interval);
  }, [tau, targetValue, stepStartTime, stepStartValue]);
  
  const handleReset = () => {
    setCurrentTime(0);
    setStepStartTime(0);
    setStepStartValue(0);
    setData([]);
  };
  
  return (
    <Card className="w-full max-w-4xl">
      <CardHeader>
        <CardTitle>First Order Step Response</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Time Constant (τ)</label>
              <input
                type="range"
                min="0.1"
                max="5"
                step="0.1"
                value={tau}
                onChange={(e) => setTau(parseFloat(e.target.value))}
                className="w-48"
              />
              <span className="ml-2">{tau.toFixed(1)}</span>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Target Value</label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={targetValue}
                onChange={(e) => setTargetValue(parseFloat(e.target.value))}
                className="w-48"
              />
              <span className="ml-2">{targetValue.toFixed(1)}</span>
            </div>
            <button
              onClick={handleReset}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Reset
            </button>
          </div>
          
          <div className="h-96">
            <LineChart
              width={800}
              height={400}
              data={data}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="time"
                label={{ value: 'Time (s)', position: 'bottom' }}
              />
              <YAxis
                domain={[0, 2]}
                label={{ value: 'Output', angle: -90, position: 'left' }}
              />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#8884d8"
                dot={false}
              />
            </LineChart>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default FirstOrderSystem;