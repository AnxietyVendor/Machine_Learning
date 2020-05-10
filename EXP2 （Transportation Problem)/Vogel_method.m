function A=Vogel_method(A,price,prod,sell,m,n)
%�þ���A���س�ʼ���䷽�����ո���ֵΪ0
    H = ones(1, m+n);%��Ǵ�[�У���]�Ƿ��������������
    while any(H) 
        C = col_row_Difference(price,m,n);
        temp = -inf;
        %�ҵ��������������Ĳ��
        for i = 1:m+n
            if temp < C(i) && C(i) ~= -inf
                temp = C(i);
                v = i; %��¼������λ��
            end
        end
        %��v������в��
        if v <= m
            price(price == -inf) = inf;
            [~,index] = min(price(v,:)); %�ҵ���С�˼ۺ��±�
            price(price == inf) = -inf;
            A(v,index) = min(sell(index),prod(v)); %ȷ������
            prod(v) = prod(v) - A(v,index); %���²���
            sell(index) = sell(index) - A(v,index);%��������
            % �жϷ���Ի���һ�У������ľ���
            if prod(v) == 0
                H(v) = 0;%�����������������
                price(v,:) = [-inf];
            end
            % �ж��Ƿ���Ի���һ�У������ľ���
            if sell(index) == 0
                H(m + index) = 0;%�����������������
                price(:,index) = [-inf];
            end
        end
        %��v������в��
        if v > m
            price(price == -inf) = inf;
            [~,index] = min(price(:,v - m)'); %��Сֵ����Ϊ-inf
            price(price == inf) = -inf;
            A(index, v - m) = min(prod(index),sell(v - m));
            prod(index) = prod(index) - A(index, v-m);
            sell(v - m) = sell(v - m) - A(index, v-m);
            if prod(index) == 0
                H(index) = 0;
                price(index,:) = [-inf];
            end
            if sell(v - m) == 0
                H(v) = 0;
                price(:,v - m) = [-inf];
            end
        end
        %����ͬ�л�ͬ��ֻ��һ��Ԫ��
        [I,J] = find(~isnan(A));
        if C == -inf
            %��������Ŀ����ʱ���0
            if size(I) ~= m+n-1
                [x,y] = finding(A);
                A(x,y) = min(prod(x),sell(y));
                H = zeros(1,m + n);
            end
        end
        
    end
        
function B = col_row_Difference(A,m,n) 
%����[�в��,�в��]
%����m*n�ľ��󣬷���1��m+n�еľ����¼�в�����в��
    
    %�����в��
    %���ҵ���С�������˼�
    for i = 1:m
        temp1 = inf;
        temp2 = inf;
        h = i;
        for j = 1:n
            if temp1 > A(i,j) && A(i,j) ~= -inf
                temp1 = A(i,j)
                h = j;%��¼��λ��
            end
        end
        flag = 0 %�ж��Ƿ����Ѱ��
        for j = 1:n
            if temp2 > A(i,j) && (j ~= h) && A(i,j) ~= -inf
                temp2 = A(i,j);
                flag = 1
            end
        end
        if flag == 1
            B(i) = temp2 - temp1;
        else
            B(i) = -inf;% ��ֻ��һ��Ԫ��ʱ���Ϊ-inf
        end
    end
    
    %�ټ����в��
    for i = 1:n
        temp1 = inf;
        temp2 = inf;
        k = i;
        for j = 1:m
            %k = 0; %��ʼ��kֵ
            if temp1 > A(j,i) && A(j,i) ~= -inf
                temp1 = A(j,i);
                k = j;%��¼��λ��
            end
        end
        flag =0;
        for j = 1:m
            if temp2 > A(j,i) && (j ~= k) && A(j,i) ~= -inf
                temp2 = A(j,i);
                flag = 1;
            end
        end
        if flag ==1
            B(m+i)=temp2-temp1;
        else
            B(m+i)=-inf;% ��ֻ��һ��Ԫ��ʱ���Ϊ-inf
        end
    end        
        
        
function [x,y] = finding(A)
%��һ�л�һ��ʣ�����һ��Ԫ��ʱ���ҵ������еķǸ�Ԫ�ز������±�
    [m,n] = size(A);
    for i = 1:m
        for j = 1:n
            if A(i,j)~=-inf
                x = i;
                y = j;
                break
            end
        end
    end        
        
        
            