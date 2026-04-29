function outputImage = enhanceContrast(inputImage, config)
    %ENHANCECONTRAST Stretch image contrast for downstream edge detection.
    
    if nargin < 2
        config = defaultConfig();
    end
    
    outputImage = imadjust(inputImage, stretchlim(inputImage, [0.01 0.99]), []);
end

