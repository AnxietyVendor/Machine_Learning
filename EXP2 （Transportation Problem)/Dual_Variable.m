function pos = Dual_Variable(A,price)
%��pos���󷵻ظ����ǻ���������λ�õ�λ��
%AΪ��ʼ��
    [m,n] = size(price);
    [I,J] = find(~isnan(A)); %�ҵ�������λ��
    b = [0;price(sub2ind([m,n],I,J))]; %λ�Ʒ��̵��Ҷ���
    UV = zeros(m+n);
    UV(1,1) = 1; %�ٶ�v1����0
    for i = 1:n+m-1
        UV(i+1, [I(i),J(i) + m]) = 1;
    end
    x = UV \ b; %��ⷽ����
    u = x(1:m);
    v = x(m + 1:end);
    [nI,nJ] = find(isnan(A)); %�ҵ��ǻ�����λ��
    pos = zeros(m,n);
    for i = 1:length(nI)
        pos(nI(i),nJ(i)) = u(nI(i)) + v(nJ(i));
    end
    
    
    
    