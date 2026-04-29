function outputImage = reduceNoise(inputImage, config)
    %REDUCENOISE Denoise an image while preserving plate edges.
    
    if nargin < 2
        config = defaultConfig();
    end
    
    medianFiltered = medfilt2(inputImage, config.preprocessing.medianFilterSize);
    outputImage = imgaussfilt(medianFiltered, config.preprocessing.gaussianSigma);
end

