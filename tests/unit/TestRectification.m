classdef TestRectification < matlab.unittest.TestCase
    methods (Test)
        function rectifyExtractsTighterTextCrop(testCase)
            cfg = defaultConfig();
            [sceneImage, meta] = createSyntheticPlateImage("BPK1234");
            oversizedBox = [meta.plateBox(1) - 35, meta.plateBox(2) - 15, ...
                meta.plateBox(3) + 70, meta.plateBox(4) + 30];

            [rectifiedPlate, rectifyMeta] = rectifyPlate(sceneImage, oversizedBox, cfg);

            testCase.verifyNotEmpty(rectifiedPlate);
            testCase.verifyNotEmpty(rectifyMeta.textOnlyBBox);
            testCase.verifyLessThan(rectifyMeta.textOnlyBBox(3), size(rectifiedPlate, 2));
            testCase.verifyLessThan(rectifyMeta.textOnlyBBox(4), size(rectifiedPlate, 1));
            testCase.verifyNotEmpty(rectifyMeta.textOnlyPlate);
        end

        function rectifyProducesRowCompositeForTwoRowPlate(testCase)
            cfg = defaultConfig();
            [sceneImage, meta] = createSyntheticTwoRowPlateImage("BPK", "1234");

            [~, rectifyMeta] = rectifyPlate(sceneImage, meta.plateBox, cfg);

            testCase.verifyEqual(string(rectifyMeta.layoutHint), "two_row");
            testCase.verifySize(rectifyMeta.rowBBoxes, [2 4]);
            testCase.verifyNotEmpty(rectifyMeta.rowCompositePlate);
        end

        function tinyTextCropIsUpscaledBeforeOcr(testCase)
            cfg = defaultConfig();
            [sceneImage, meta] = createSyntheticTinyPlateImage("WVU8899");

            [~, rectifyMeta] = rectifyPlate(sceneImage, meta.plateBox, cfg);

            testCase.verifyNotEmpty(rectifyMeta.textOnlyPlate);
            testCase.verifyGreaterThanOrEqual(size(rectifyMeta.textOnlyPlate, 1), ...
                cfg.rectification.minTextHeightPixels);
        end

        function rectifyAddsMoreRightTextPaddingThanLeft(testCase)
            cfg = defaultConfig();
            mask = false(60, 180);
            mask(18:42, 50:120) = true;

            bbox = localTestExpandTextBox(mask, cfg);

            testCase.verifyLessThanOrEqual(bbox(1), 48);
            testCase.verifyGreaterThanOrEqual(bbox(1) + bbox(3), 128);
        end
    end
end

function bbox = localTestExpandTextBox(mask, cfg)
    stats = regionprops("table", mask, "BoundingBox");
    rawBox = stats.BoundingBox(1, :);
    leftRatio = double(cfg.rectification.textPaddingLeftRatio);
    rightRatio = double(cfg.rectification.textPaddingRightRatio);
    verticalRatio = double(cfg.rectification.textPaddingVerticalRatio);

    x1 = max(1, floor(rawBox(1) - rawBox(3) * leftRatio));
    y1 = max(1, floor(rawBox(2) - rawBox(4) * verticalRatio));
    x2 = min(size(mask, 2), ceil(rawBox(1) + rawBox(3) + rawBox(3) * rightRatio));
    y2 = min(size(mask, 1), ceil(rawBox(2) + rawBox(4) + rawBox(4) * verticalRatio));
    bbox = [x1 y1 max(1, x2 - x1) max(1, y2 - y1)];
end
