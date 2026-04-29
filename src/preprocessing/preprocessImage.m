function [preprocessedImage, metadata] = preprocessImage(inputImage, config)
    %PREPROCESSIMAGE Convert to grayscale, reduce noise, and enhance contrast.
    
    if nargin < 2
        config = defaultConfig();
    end
    
    config = validateConfig(config);
    
    if isempty(inputImage)
        error("preprocessImage:EmptyInput", "Input image is empty.");
    end
    
    grayImage = im2gray(inputImage);
    grayImage = im2uint8(grayImage);
    normalizedImage = normalizeLighting(grayImage, config);
    denoisedImage = reduceNoise(normalizedImage, config);
    
    contrastMethod = lower(string(config.preprocessing.contrastMethod));
    switch contrastMethod
        case "imadjust"
            preprocessedImage = enhanceContrast(denoisedImage, config);
        case "adapthisteq"
            preprocessedImage = adapthisteq(denoisedImage, ...
                "NumTiles", config.preprocessing.claheTiles, ...
                "ClipLimit", config.preprocessing.claheClipLimit);
        otherwise
            preprocessedImage = enhanceContrast(denoisedImage, config);
    end
    
    metadata = struct( ...
        "grayImage", grayImage, ...
        "normalizedImage", normalizedImage, ...
        "denoisedImage", denoisedImage, ...
        "preprocessedImage", preprocessedImage, ...
        "contrastMethod", contrastMethod);
end
