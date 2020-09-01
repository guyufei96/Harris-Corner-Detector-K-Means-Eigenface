ori_1 = imread('mandm.png');
ori_2 = imread('peppers.png');

gray1 = rgb2gray(ori_1);
gray2 = rgb2gray(ori_2);


[m1 n1 z1] = size(ori_1);
[m2 n2 z2] = size(ori_2);



k = 5;
R1 = reshape(ori_1(:,:,1),m1*n1,1);
G1 = reshape(ori_1(:,:,2),m1*n1,1);
B1 = reshape(ori_1(:,:,3),m1*n1,1);

R2 = reshape(ori_2(:,:,1),m2*n2,1);
G2 = reshape(ori_2(:,:,2),m2*n2,1);
B2 = reshape(ori_2(:,:,3),m2*n2,1);

vector1 = [R1 G1 B1];
vector1 = double(vector1); 

vector2 = [R2 G2 B2];
vector2 = double(vector2); 

J1 = my_kmeans_function(vector1,k);
I1 = label2rgb(reshape(J1,m1,n1));

J2 = my_kmeans_function(vector2,k);
I2 = label2rgb(reshape(J2,m2,n2));

subplot(2,2,1), imshow(ori_1), title('mandm');
subplot(2,2,2),imshow(I1), title('mandm-kmeans');
subplot(2,2,3), imshow(ori_2), title('peppers');
subplot(2,2,4), imshow(I2), title('peppers-kmeans');

