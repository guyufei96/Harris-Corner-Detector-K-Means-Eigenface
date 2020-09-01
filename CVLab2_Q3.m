close all;
clear all;

%read all traning images
train_num = 144;
img_dims = [231 195];

trainpath = 'Yale-FaceA\trainingset\';
testpath = 'Yale-FaceA\testset\';

train_filenames = dir([trainpath '*.png']);    % return a structure with filenames
test_filenames = dir([testpath '*.png']);      

prompt = {'Choose the test face (a number between 1 to 11):'};
dlg_title = 'Input of PCA-Based Face Recognition System';
num_lines= 1;
def = {'1'};
Test_addr_input  = inputdlg(prompt,dlg_title,num_lines,def);
number = str2num(cell2mat(Test_addr_input));
Test_face_addr = [testpath test_filenames(number).name];

    

data_train = [];
for i = 1 : train_num
    filename = [trainpath train_filenames(i).name];   % filename in the list
    train_face = imread(filename);
    vec = reshape(train_face,231*195,1); %transfer to N*1 vector
    data_train = [data_train vec]; %store in the dataset 
end
data_train = double(data_train);

%find the mean face
mean_face = mean(data_train,2);
show_meanface = uint8(reshape(mean_face,231,195));

%substract it from origin face
A1 = data_train;
for i = 1 : train_num
    A1(:,i) = data_train(:,i) - mean_face;
end

A1 = double(A1);
[A1_row,A1_col] = size(A1);

%Peform PCA on the data matrix
C = (1/train_num) * A1' * A1;
[C_row,C_col] = size(C);

%calculate the top K eigenvectors and eigenvalues
%K = 10;
K = 15;
[V,D] = eigs(C,K);

%Compute the eigenfaces
eigenvalues = [];
for i = 1 : K
    mv = A1 * V(:,i);
    mv = mv/norm(mv);
    eigenvalues = [eigenvalues mv];
    [eh,ew] = size(eigenvalues);
end

%Project each training image onto the new space
img_project = [];
for i = 1:train_num
    temp = double(A1(:,i)') * eigenvalues ;
    img_project = [img_project temp'];
end


%read the test image
test_face = imread(Test_face_addr);
[m,n] = size(test_face);
temp_test_face = reshape(test_face,m*n,1);
temp_test_face = double(temp_test_face) - mean_face;
%calculate the similarity of the input to each training image
feature_vec = temp_test_face' * eigenvalues ;

dist = [];
for i = 1 : train_num
    distance = norm(feature_vec' - img_project(:,i))^2;
    dist = [dist distance];
end

[dist_min index] = sort(dist);
num1 = index(1);
num2 = index(2);
num3 = index(3);

img1 = data_train(:,num1);
img1 = reshape(uint8(img1),231,195);
img2 = data_train(:,num2);
img2 = reshape(uint8(img2),231,195);
img3 = data_train(:,num3);
img3 = reshape(uint8(img3),231,195);

%Display the 15 eigenfaces and recognition
figure;

for i = 1:K
    im = eigenvalues(:,i);
    im = reshape(im,231,195);
    subplot(5,5,i);
    im = imagesc(im);colormap('gray');
end
subplot(5,5,K+6), imshow(test_face), title('Test image');
subplot(5,5,K+7),imshow(img1), title('Recognition image1');
subplot(5,5,K+8), imshow(img2), title('Recognition image2');
subplot(5,5,K+9), imshow(img3), title('Recognition image3');
subplot(5,5,K+1), imshow(show_meanface), title('mean face');

