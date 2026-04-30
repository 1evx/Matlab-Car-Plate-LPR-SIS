function preparedImage = prepareTesseractPlateImage(plateImage, config)
    %PREPARETESSERACTPLATEIMAGE Prepare a plate crop for OCR.

    if nargin < 2
        config = struct();
    end

    if isempty(plateImage)
        preparedImage = uint8([]);
        return;
    end

    preparedImage = plateImage;
    if ndims(preparedImage) == 3
        preparedImage = im2gray(preparedImage);
    end

    preparedImage = im2single(preparedImage);
    preparedImage = imadjust(preparedImage);
    claheTiles = localClaheTileCount(size(preparedImage));
    if all(claheTiles >= 1)
        preparedImage = adapthisteq(preparedImage, "NumTiles", claheTiles, "ClipLimit", 0.02);
    end
    preparedImage = imsharpen(preparedImage, "Radius", 1.2, "Amount", 0.8);

    scaleFactor = 2.0;
    if isfield(config, "classification") && isfield(config.classification, "tesseractScaleFactor")
        scaleFactor = double(config.classification.tesseractScaleFactor);
    end
    scaleFactor = max(1, scaleFactor);
    preparedImage = imresize(preparedImage, scaleFactor, "bicubic");
    preparedImage = im2uint8(preparedImage);
end

function claheTiles = localClaheTileCount(imageSize)
    imageHeight = imageSize(1);
    imageWidth = imageSize(2);

    % Keep CLAHE enabled for small crops, but avoid requesting more tiles
    % than the image can meaningfully support.
    maxTileRows = max(1, floor(double(imageHeight) / 2));
    maxTileCols = max(1, floor(double(imageWidth) / 2));
    claheTiles = [min(8, maxTileRows) min(8, maxTileCols)];
end
