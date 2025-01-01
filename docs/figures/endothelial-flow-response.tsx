import React, { useState, useEffect } from 'react';
import { Slider } from '@/components/ui/slider';
import { Alert, AlertDescription } from '@/components/ui/alert';

const EndothelialFlowResponse = () => {
  const [actinDensity, setActinDensity] = useState(50);
  const [shearStressLevel, setShearStressLevel] = useState(1); // 0-2 representing Low-High
  const [cellStability, setCellStability] = useState('stable');
  
  const getShearStressLabel = (level) => {
    switch (level) {
      case 0:
        return 'Low';
      case 1:
        return 'Medium';
      case 2:
        return 'High';
      default:
        return 'Medium';
    }
  };

  useEffect(() => {
    // Calculate cell stability based on actin density and shear stress
    if (shearStressLevel === 2) {  // High shear stress
      if (actinDensity < 30) {
        setCellStability('rupture');
      } else if (actinDensity > 70) {
        setCellStability('rigid');
      } else {
        setCellStability('stable');
      }
    } else if (shearStressLevel === 0) {  // Low shear stress
      if (actinDensity < 20) {
        setCellStability('unstable');
      } else if (actinDensity > 80) {
        setCellStability('rigid');
      } else {
        setCellStability('stable');
      }
    } else {  // Medium shear stress
      if (actinDensity < 25) {
        setCellStability('unstable');
      } else if (actinDensity > 75) {
        setCellStability('rigid');
      } else {
        setCellStability('stable');
      }
    }
  }, [actinDensity, shearStressLevel]);

  const getCellColor = () => {
    switch (cellStability) {
      case 'rupture':
        return 'bg-red-500';
      case 'unstable':
        return 'bg-yellow-500';
      case 'rigid':
        return 'bg-blue-500';
      default:
        return 'bg-green-500';
    }
  };

  const getStatusMessage = () => {
    const stressLevel = getShearStressLabel(shearStressLevel);
    switch (cellStability) {
      case 'rupture':
        return `Warning: Cell at risk of detachment under ${stressLevel} shear stress conditions`;
      case 'unstable':
        return 'Cell adhesion compromised - low actin density affecting structural integrity';
      case 'rigid':
        return 'Excessive actin formation may impair cell adaptability';
      default:
        return `Cell stable - optimal actin density for ${stressLevel} shear stress conditions`;
    }
  };

  return (
    <div className="w-full max-w-2xl p-6 space-y-6">
      <h2 className="text-2xl font-bold mb-4">Endothelial Cell Shear Response</h2>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">
            Actin Density: {actinDensity}%
          </label>
          <Slider
            value={[actinDensity]}
            onValueChange={(value) => setActinDensity(value[0])}
            max={100}
            min={0}
            step={1}
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">
            Shear Stress Level: {getShearStressLabel(shearStressLevel)}
          </label>
          <Slider
            value={[shearStressLevel]}
            onValueChange={(value) => setShearStressLevel(value[0])}
            max={2}
            min={0}
            step={1}
          />
        </div>
      </div>

      <div className="relative h-64 border rounded-lg overflow-hidden bg-gray-50">
        {/* Flow direction indicator */}
        <div className="absolute top-4 left-4 flex items-center">
          <span className="mr-2">Flow Direction</span>
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
          </svg>
        </div>

        {/* Cell visualization */}
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
          <div className={`w-32 h-32 rounded-full ${getCellColor()} transition-colors duration-300 
            flex items-center justify-center relative`}>
            <div className="w-12 h-12 rounded-full bg-gray-700 absolute" 
              style={{
                transform: `translateX(${shearStressLevel * 10}px)`
              }}
            />
          </div>
        </div>
      </div>

      <Alert>
        <AlertDescription>
          {getStatusMessage()}
        </AlertDescription>
      </Alert>

      <div className="mt-4 text-sm">
        <h3 className="font-medium mb-2">Cell Response Analysis:</h3>
        <ul className="list-disc pl-5 space-y-1">
          <li>Shear stress level: {getShearStressLabel(shearStressLevel)}</li>
          <li>Actin network density: {actinDensity}%</li>
          <li>Cell status: {cellStability}</li>
        </ul>
      </div>
    </div>
  );
};

export default EndothelialFlowResponse;