function showPreprocessingDebugFigure(preprocessMeta)
%SHOWPREPROCESSINGDEBUGFIGURE Show preprocessing outputs in a standalone figure.

    figureHandle = findall(groot, "Type", "figure", "Tag", "LPRPreprocessingDebug");
    if isempty(figureHandle) || ~isvalid(figureHandle(1))
        figureHandle = figure( ...
            "Name", "LPR Preprocessing Debug", ...
            "NumberTitle", "off", ...
            "Tag", "LPRPreprocessingDebug");
    else
        figureHandle = figureHandle(1);
        clf(figureHandle);
        figure(figureHandle);
    end

    tiledlayout(figureHandle, 2, 2, "TileSpacing", "compact", "Padding", "compact");

    nexttile;
    imshow(preprocessMeta.grayImage);
    title("Grayscale");

    nexttile;
    imshow(preprocessMeta.normalizedImage);
    title("Normalized");

    nexttile;
    imshow(preprocessMeta.denoisedImage);
    title("Denoised");

    nexttile;
    imshow(preprocessMeta.preprocessedImage);
    title("Preprocessed");
end
