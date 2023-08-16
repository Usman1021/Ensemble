%% Global Motion estimation learning
% Input for training, development and testing set videos
x = dir('*.mp4');
[length temp] = size(x);
for k=1:length
    videoname = x(k).name;
    videoFileReader = vision.VideoFileReader(videoname);

    numFrames = 0;
    while ~isDone(videoFileReader)
        step(videoFileReader);
        numFrames = numFrames + 1;
    end

    reset(videoFileReader);                   
    movMean = step(videoFileReader);
    imgB = movMean;
    imgBp = imgB;
    correctedMean = imgBp;
    % set the video length 
    range = 40:40:numFrames;
    Hcumulative = eye(3);
    for i=1:size(range,2)
        ii=range(i);
        ref=ii;
        while ~isDone(videoFileReader) && ii < ref + 40
            imgA = imgB; 
            imgAp = imgBp;
            imgB = step(videoFileReader);
         % Estimate transform from frame A to frame B, and fit as an s-R-t
            H = cvexEstStabilizationTform1(imgA,imgB);
            HsRt = cvexTformToSRT(H);
            Hcumulative = HsRt * Hcumulative;
            img = imwarp(imgB,rigid2d(Hcumulative),'OutputView',imref2d(size(imgB)));
             correctedMean = correctedMean + img;
            ii = ii+1;
        end
        correctedMean = correctedMean/(40);
          alpha = 0.1;
           blended = alpha *  correctedMean + (1 - alpha) * imgB;
        imwrite(blended,strcat( string(k), string(i),'.jpg'));
    end
end
%%
function H = cvexEstStabilizationTform1(leftI,rightI,ptThresh)
ptThresh = 0.1;
leftI = im2gray(leftI);
rightI = im2gray(rightI);

pointsA = detectFASTFeatures(leftI, 'MinContrast', ptThresh);
pointsB = detectFASTFeatures(rightI, 'MinContrast', ptThresh);
% Extract FREAK descriptors for the corners
[featuresA, pointsA] = extractFeatures(leftI, pointsA);
[featuresB, pointsB] = extractFeatures(rightI, pointsB);
indexPairs = matchFeatures(featuresA, featuresB);
pointsA = pointsA(indexPairs(:, 1), :);
pointsB = pointsB(indexPairs(:, 2), :);
% Estimating Transform
[tform, ~, ~, status] = estimateGeometricTransform(pointsB, pointsA, 'rigid');
H = tform.T;
end
%%
function [H,s,ang,t,R] = cvexTformToSRT(H)
R = H(1:2,1:2);
t = H(3, 1:2);
ang = mean([atan2(R(2),R(1)) atan2(-R(3),R(4))]);
s = mean(R([1 4])/cos(ang));
% Reconstitute new s-R-t transform:
R = [cos(ang) -sin(ang); sin(ang) cos(ang)];
H = [[s*R; t], [0 0 1]'];
end