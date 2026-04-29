function outputImage = normalizeLighting(inputImage, config)
    %NORMALIZELIGHTING Normalize local illumination variations.
    
    if nargin < 2
        config = defaultConfig();
    end
    
    gray = im2uint8(inputImage);
    outputImage = adapthisteq(gray, ...
        "NumTiles", config.preprocessing.claheTiles, ...
        "ClipLimit", config.preprocessing.claheClipLimit);
end

