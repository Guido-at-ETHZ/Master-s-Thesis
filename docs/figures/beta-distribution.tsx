import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';

const BetaDistribution = () => {
  const [alpha, setAlpha] = useState(2);
  
  // Function to calculate log gamma
  const logGamma = (z) => {
    const p = [
      676.5203681218851,
      -1259.1392167224028,
      771.32342877765313,
      -176.61502916214059,
      12.507343278686905,
      -0.13857109526572012,
      9.9843695780195716e-6,
      1.5056327351493116e-7
    ];
    
    if (z < 0.5) {
      return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
    }
    
    z -= 1;
    let x = 0.99999999999980993;
    for (let i = 0; i < p.length; i++) {
      x += p[i] / (z + i + 1);
    }
    
    const t = z + p.length - 0.5;
    return Math.log(2 * Math.PI) / 2 + Math.log(x) - t + (z + 0.5) * Math.log(t);
  };

  // Function to calculate Beta PDF
  const betaPDF = (x, alpha, beta) => {
    if (x <= 0 || x >= 1) return 0;
    
    const logBeta = logGamma(alpha) + logGamma(beta) - logGamma(alpha + beta);
    const logPDF = (alpha - 1) * Math.log(x) + (beta - 1) * Math.log(1 - x) - logBeta;
    
    return Math.exp(logPDF);
  };

  // Generate x values
  const xValues = Array.from({ length: 200 }, (_, i) => i / 199);

  // Generate data points
  const data = xValues.map(x => ({
    x,
    'Beta Distribution': betaPDF(x, alpha, alpha)
  }));

  const updateAlpha = (value) => {
    setAlpha(value[0]);
  };

  // Function to describe the shape based on alpha value
  const getShapeDescription = (alpha) => {
    if (alpha < 1) return "U-shaped (both tails go to infinity)";
    if (alpha === 1) return "Uniform distribution";
    if (alpha < 2) return "Broad, bathtub-like shape";
    if (alpha === 2) return "Parabolic shape";
    if (alpha < 4) return "Rounded, bell-like shape";
    if (alpha < 7) return "Increasingly peaked bell shape";
    return "Highly concentrated, near-normal distribution";
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Symmetric Beta Distribution (α = β)</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-8">
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>α = β = {alpha.toFixed(2)}</Label>
              <Slider 
                value={[alpha]}
                min={0.1}
                max={10}
                step={0.1}
                onValueChange={updateAlpha}
              />
            </div>
            <div className="text-sm text-muted-foreground">
              Shape: {getShapeDescription(alpha)}
            </div>
          </div>
          
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="x" 
                  label={{ value: 'x', position: 'bottom' }}
                  domain={[0, 1]}
                />
                <YAxis 
                  label={{ value: 'Probability Density', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="Beta Distribution" 
                  stroke="#2563eb" 
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default BetaDistribution;