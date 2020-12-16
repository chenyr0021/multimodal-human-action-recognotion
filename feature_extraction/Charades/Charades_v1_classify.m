function [rec_all,prec_all,ap_all,map]=Charades_v1_classify(clsfilename,gtpath)
%
%     Input:    clsfilename: path of the input file
%                    gtpath: the path of the groundtruth file
%
%    Output:        rec_all: recall
%                  prec_all: precision
%                    ap_all: AP for each class
%                       map: MAP 
%
% Example:
%
%  [rec_all,prec_all,ap_all,map]=Charades_v1_classify('test_submission_classify.txt','Charades_v1_test.csv');
%
% Code adapted from THUMOS15 
%

[gtids,gtclasses] = load_charades(gtpath); % ����ground truth�ļ�
nclasses = 157;
ntest = length(gtids);

% load test scores
[testids,testscores]=textread(clsfilename,'%s%[^\n]'); 
nInputNum=size(testscores,1);
if nInputNum<ntest
    fprintf('Warning: %d Videos missing\n',ntest-nInputNum);
end
for i=1:nInputNum
    id = testids{i};
	% �ж�ÿ�е�testscores�����Ƿ���������157
    z=regexp(testscores{i},'\t','split'); % ʹ��������ʽ�ָ�
    eleNum=size(z,2);
    if eleNum~=nclasses&&eleNum~=nclasses+1
        z=regexp(testscores{i},' ','split');
    end
    eleNum=size(z,2);
    if eleNum~=nclasses&&eleNum~=nclasses+1
        fprintf('Error: Incompatible number of classes\n');
    end
	% ȥ���ո��\t
    for j=1:eleNum
        z{j}=regexprep(z{j},'\t',''); % ʹ��������ʽ�滻'\t'
        z{j}=regexprep(z{j},' ','');
    end
	% str to double
    x = zeros(nclasses,1); 
    for j=1:nclasses
		x(j) = str2double(z{j});
    end
    testscores{i} = x;
end
predictions = containers.Map(testids,testscores);
groundtruth = containers.Map(gtids, gtclasses);

% compare test scores to ground truth
% gtlabel = zeros(ntest,nclasses);
% test = -inf(ntest,nclasses);
% for i=1:ntest
    % id = gtids{i};
    % gtlabel(i,gtclasses{i}+1) = 1; 
    % if predictions.isKey(id)
	% test(i,:) = predictions(id);
    % end
% end

% �����ֵ�����ֵ��id����һһ��Ӧ�ģ�������һ����id��ȱʧ�����ã�
gtlabel = zeros(nInputNum, nclasses);
test = -inf(nInputNum, nclasses);
for i=1:nInputNum % ����testidѰ�Ҷ�Ӧ����ֵ
	id = testids{i};
    gt = groundtruth(id);
    if length(gt) ~= 0
        gtlabel(i,gt+1) = 1; 
    end
	test(i,:) = predictions(id);	
end
 

for i=1:nclasses
    [rec_all(:,i),prec_all(:,i),ap_all(:,i)]=THUMOSeventclspr(test(:,i),gtlabel(:,i));
end
map=mean(ap_all);
wap=sum(ap_all.*sum(gtlabel,1))/sum(gtlabel(:));
fprintf('\n\n')
fprintf('MAP: %f\n',map);
fprintf('WAP: %f (weighted by size of each class)',wap);
fprintf('\n\n')


function [rec,prec,ap]=THUMOSeventclspr(conf,labels)
[so,sortind]=sort(-conf);
tp=labels(sortind)==1;
fp=labels(sortind)~=1;
npos=length(find(labels==1));

% compute precision/recall
fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/npos;
prec=tp./(fp+tp);

% compute average precision

ap=0;
tmp=labels(sortind)==1;
for i=1:length(conf)
    if tmp(i)==1
        ap=ap+prec(i);
    end
end
ap=ap/npos;


function [gtids,gtclasses] = load_charades(gtpath)
f = fopen(gtpath);

% read column headers
headerline = textscan(f,'%s',1);
headerline = regexp(headerline{1}{1},',','split');
ncols = length(headerline);
headers = struct();
for i=1:ncols
    headers = setfield(headers,headerline{i},i);
end

% read data
gtcsv = textscan(f,repmat('%q ',[1 ncols]),'Delimiter',',');
ntest = size(gtcsv{1},1);
gtids = cell(ntest,1);
gtclasses = cell(ntest,1);
for i=1:ntest
    id = gtcsv{headers.id}{i};
    classes = gtcsv{headers.actions}{i};
    % ��������actionΪ�գ���¼id������������idΪ0��0 double
    if length(classes)==0
        gtclasses{i} = [];
        gtids{i} = id;
        continue; 
    end
    classes = regexp(classes,';','split');
    for j=1:length(classes)
        tmp = regexp(classes{j},' ','split');
        [class,s,e] = tmp{:};
        classes{j} = str2double(class(2:end));
    end
    gtids{i} = id;
    gtclasses{i} = cell2mat(classes);
end


