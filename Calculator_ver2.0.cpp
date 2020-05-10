#include<stdio.h>
#include<stdlib.h> 
#include<math.h>
#include<string.h>

#define TRUE 1
#define FALSE 0
#define OK 1
#define ERROR 0
#define INFEASIBLE -1
//#define OVERFLOW -2 

typedef int ElemType; 
typedef int Status;

//�������Ա�����������������
#define LIST_INIT_SIZE 100//����ռ��ʼ������������Ԫ�ظ����� 
#define LISTINCREMENT 10 //����ռ�������� ������Ԫ�ظ�����
typedef struct{
	
	ElemType *elem;//����ռ������ַ
	int length;//���е�ǰ����Ԫ�ظ��� 
	int listsize;//��ǰ����Ĵ洢������sizeof��ElemType����
		 
}SqList,Vector,*vector;

//������ʼ��ʵ�� 
void Init_Vector(Vector &V, int dim){//������������������ά��dim 
 	V.elem = (ElemType*)malloc(LIST_INIT_SIZE * sizeof(ElemType));
 	if(!V.elem) exit(OVERFLOW); 
 	V.length = dim;//��ʼ������Ԫ�ظ��� 
 	V.listsize = LIST_INIT_SIZE * sizeof(ElemType);//��ʼ����ǰ�������� 
 	//V.col = type;//��ʼ���������� 
	 for(int i = 0;i < dim;i++) {
 		scanf("%d",&V.elem[i]);
	}	 
 } 
 
 void Print_Vector(Vector &V){
 	if(V.length == 0) printf("Invalid vector to print"); 
 	for(int i = 0;i < V.length;i++) printf(" %d",V.elem[i]);
 	//for(int i = 0;i <= V.length - 1;i++) printf("%f ",V.elem[i]);
 	putchar('\n');
 } 

//������ֵ��0ʵ�� 
void Vector_clear(Vector &V){//������ֵ��Ϊ0���� 
	if(V.elem == NULL) printf("vector doesn't exist");
	else{
		for(int i = 0;i < V.length;i++){
		V.elem[i] = 0;	
		}
	} 
}

//����ɾ��ʵ�� 
void Vector_delete(Vector &V){//�ͷ�������ռ�ռ�
	ElemType* p = V.elem;
	V.length = 0;
	V.listsize = 0;
	free(p); 
} 

//�����ӷ�ʵ�� 
vector Vector_add(vector va, vector vb){
	if(!va->elem || !vb->elem) printf("vectors don't exist");
	//if(va->col != vb->col) printf("invalid addition��different vector type");//��������������������� 
	if(va->length != vb->length) printf("invalid addition: different vector dimension"); //��ͬά�����������
	ElemType* v1;
	ElemType* v2;
	vector vs;
	v1 = va->elem;//����va�����ַ 
	v2 = vb->elem;//����vb�����ַ
	vs = va;
	for(int i = 0;i < va->length;i++){
		*(v1 + i) += *(v2 + i);
	}
	return vs;
}

//��������ʵ��
vector Vector_subtract(vector va, vector vb){
	if(!va->elem || !vb->elem) printf("vectors don't exist");
	//if(va->col != vb->col) printf("invalid addition��different vector type");//��������������������� 
	if(va->length != vb->length) printf("invalid addition: different vector dimension"); //��ͬά�����������
	ElemType* v1;
	ElemType* v2;
	vector vs;
	v1 = va->elem;//����va�����ַ 
	v2 = vb->elem;//����vb�����ַ
	vs = va;
	for(int i = 0;i < va->length;i++){
		*(v1 + i) -= *(v2 + i);
	}
	return vs;
} 

//�������ʵ�� 
vector Vector_dot_multiply(vector va, vector vb){
	if(!va->elem || !vb->elem) printf("vectors don't exist");
	if(va->length != vb->length) printf("invalid addition: different vector dimension"); //��ͬά�����������
	ElemType* v1;
	ElemType* v2;
	vector vs;
	v1 = va->elem;//����va�����ַ 
	v2 = vb->elem;//����vb�����ַ
	vs = va;//va����˷����
	for(int i = 0;i < va->length;i++){
		*(v1 + i) = *(v1 + i) * *(v2 + i);
	}//�����������Ӧ����֮�� 
	return vs; 
	
} 

//����ģ������
double Vector_module(vector v){
	double module = 0;
	vector product = Vector_dot_multiply(v,v);//����������˵Ļ�����
	ElemType* vt = product->elem;
	for(int i = 0;i < v->length;i++){
		module += *(vt + i);//�������������֮�� 
	} 
	return sqrt(abs(module)); 
} 

//�����н�����ֵ����
double Vector_cos(vector va, vector vb) {
	double cos = 0;
	double mod_va = Vector_module(va);
	double mod_vb = Vector_module(vb);
	
	vector vavb = Vector_dot_multiply(va,vb);
	double mod_vavb = Vector_module(vavb);
	
	return mod_vavb/(mod_va * mod_vb);
}
 

//����˳��������Ķ���ʽ������ 
//---------------------------------------------------------------------------------------------------------------------------
//����Ԫ�ر�ʾ
typedef struct{
	//int coef; //��ϵ������ʽ
	float coef; //ʵϵ������ʽ 
	int expn;
}datatype;

typedef struct{
	
	datatype* term;//����ռ������ַ
	int length;//���е�ǰ����ʽ���� 
	int listsize;//��ǰ����Ĵ洢����
 
}ArrayPoly,*Polyptr;//�������ú�ָ��

void Init_ArrayPoly(ArrayPoly &poly, int terms){//�������ʽ������ʼ������ʽ 
	 poly.term = (datatype*)malloc(LIST_INIT_SIZE * sizeof(datatype));
	 if(!poly.term) exit(OVERFLOW);
	 poly.length = terms;//��¼����ʽ���� 
	 poly.listsize = LIST_INIT_SIZE * sizeof(ElemType);//��¼�ѷ���洢����
	 for(int i = 0;i < terms;i++){
	 	scanf("%f %d",&poly.term[i].coef,&poly.term[i].expn);
	 } 
} 

//������Ա�����ʽ 
void Print_ArrayPoly(ArrayPoly &poly){
	if(!poly.term)
		printf("������Ϊ��\n");
	int i;
	double calculation;
	for(i = 0;i < poly.length;i++) calculation += poly.term[i].coef;
	
	if(fabs(calculation) < 1e-5){
		printf("0");
	}
	else{
	
	for(i = 0;i < poly.length - 1;i++){
		if(poly.term[i].expn == 0) printf("%.3f + ",poly.term[i].coef);
		if(poly.term[i].expn > 0){
		if(poly.term[i].coef >= 0)
			printf("%.3fx^%d + ",poly.term[i].coef,poly.term[i].expn);
		else
			printf("%.3fx^%d ",poly.term[i].coef,poly.term[i].expn);			
		}
	}
	if(poly.term[i].expn > 0) printf("%.3fx^%d\n",poly.term[i].coef,poly.term[i].expn);
	if(poly.term[i].expn == 0) printf("%.3f ",poly.term[i].coef);
	} 
}

//����ʽ�ӷ�ʵ�� 
Polyptr Add_ArrayPoly(Polyptr p1, Polyptr p2){//������������ʽ�Ĵ洢��ַ 
	int void_count = 0;
	datatype* pa = p1->term;
	datatype* pb = p2->term;
	datatype* media = (datatype*)malloc((p1->length+p2->length)*sizeof(datatype));
	datatype* pc = media;//ָ��ָ��͵Ĵ洢����ַ
	
	datatype* pa_last = p1->term + p1->length -1;//��¼���Ա�ĩλԪ�ص�ַ 
	datatype* pb_last = p2->term + p2->length -1;
	while(pa <= pa_last && pb <= pb_last){
		
		if(pa->expn == pb->expn){//����ͬ����
			if(pa->coef + pb->coef == 0) void_count++;
			else{
				pc->coef = pa->coef + pb->coef;
				pc->expn = pa->expn;
				pa++;
				pb++;
				pc++;		
			} 
		}
		
		if(pa->expn < pb->expn){
			pc->coef = pa->coef;
			pc->expn = pa->expn;
			pa++;
			pc++;
		}
		
		if(pb->expn < pa->expn){
			pc->coef = pb->coef;
			pc->expn = pb->expn;
			pb++;
			pc++;					
		}
		
	
	} 
	while(pa <= pa_last){//pb��ɱ���
	pc->expn = pa->expn;
	pc->coef = pa->coef;
	pa++;
	pc++; 
	} 

	while(pb <= pb_last){//pa��ɱ���
	pc->expn = pb->expn;
	pc->coef = pb->coef;
	pb++;
	pc++; 
	} 	
	
	p1->term = media;
	p1->length = p1->length + p2->length - void_count ;
	
	return p1;//����sum�ĵ�ַ 
} 

//����ʽ����ʵ�� 
Polyptr Subtract_ArrayPoly(Polyptr p1, Polyptr p2){
	int i;
	for(i = 0;i < p2->length;i++){
		p2->term[i].coef = -(p2->term[i].coef);
	}
	Polyptr minus = Add_ArrayPoly(p1,p2);
	return minus;
}

//�����ǰ����ʽ 
Polyptr Clone_ArrayPoly(Polyptr p){
	Polyptr clone = (ArrayPoly*)malloc(sizeof(ArrayPoly));	
	for(int i = 0;i<p->length;i++){
		clone->term[i] = p->term[i];
	}
	
	clone->length = p->length;
	clone->listsize = p->listsize;
	return clone;
}


//����ʽ�˷�ʵ�� 
Polyptr Multiply_ArrayPoly(Polyptr p1, Polyptr p2){
	int i;
	datatype* pa = p1->term;//ָ����������ʽ�Ĵ洢��ַ 
	datatype* pb = p2->term;
	datatype* pa_last = p1->term + p1->length - 1;
	datatype* pb_last = p2->term + p2->length - 1;
	Polyptr sum_ptr = (ArrayPoly*)malloc(sizeof(ArrayPoly));//ָ��Ͷ���ʽ��ָ�� 
	sum_ptr->length = 0;
	
	Polyptr pc = (ArrayPoly*)malloc(sizeof(ArrayPoly));
	pc->term = (datatype*)malloc((p1->length) * sizeof(datatype));

	while(pb<=pb_last){
		pc->length = p1->length;
		datatype* pcc = pc->term;//pccָ��term 
		pa = p1->term;	
		while(pa<=pa_last){
			pcc->coef = pa->coef * pb->coef;
			pcc->expn = pa->expn + pb->expn;
			pa++;
			pcc++;
		}
		sum_ptr = Add_ArrayPoly(sum_ptr,pc);
		pb++;
	}
	sum_ptr->term[sum_ptr->length].coef = 0;
	sum_ptr->term[sum_ptr->length].expn = -1;
	return sum_ptr;
}


//����ʽ��ʵ�� 
Polyptr Diff_ArrayPoly(Polyptr p, int degree){
	if(!degree) return p;
	int i,j;
	for(i = 0;i < degree;i++){
		for(j = 0;j < p->length;j++){
			p->term[j].coef = p->term[j].expn * p->term[j].coef;
			if(p->term[j].expn == 0){
				p->term[j].coef = 0;
			}
			p->term[j].expn --;				
			}	
		}
	return p;
}


 

//�����������������Ķ���ʽ������
//---------------------------------------------------------------------------------------------------------------------------

//����ʽ���浥Ԫ
typedef struct Pnode{
	datatype data;
	struct Pnode *next;
}Pnode,*Link_List;

//����ʽ��ʾΪ��ͷ���ĵ����� 
typedef Link_List polynominal;

//--------------------------------------����ʽ��������--------------------------------// 
//����ccompare�����ķ���ֵ 
#define a_e_b 0  //a = b
#define a_g_b 1  //a > b
#define a_s_b -1 //a < b
int compare(datatype a, datatype b)
{
    if(a.expn == b.expn)
        return a_e_b;
    if(a.expn > b.expn)
        return a_g_b;
    if(a.expn < b.expn)
        return a_s_b;
}

//����polyn_locate�����ķ������� 
typedef struct{
    polynominal p;//ָ�����ʽĳ��Ľ�� 
    int type;//�жϲ���Ԫ�������ʽ��Ĺ�ϵ 
}locate;

//����poly_locate�����ķ��������е�typeֵ
#define prior -1  //Ҫ��Ԫ��ֵ��������С�������е�ĳһ��㣬������һ�ڵ��ǰ�� 
#define curor 0   //Ҫ��Ԫ�ش���
#define nextor 1  //Ҫ��Ԫ��ֵ�������Ҵ��������еĽ�㣬��� 

locate polyn_locate(polynominal HEAD, datatype e, int(*compare)(datatype, datatype))//����ָ�� 
{
    locate ptr; 
    ptr.p = HEAD;
    while((ptr.p) -> next)//��ptr.p���������е����н�� 
    {
        if(compare( (ptr.p -> next -> data), e) == a_e_b)
        {//���ptr.p�ĺ��Ԫ�������ֵ��ȣ����غ��Ԫ��ָ�� 
            ptr.p = ptr.p -> next;
            ptr.type = curor;
            return ptr;
        }
        if(compare( ptr.p -> next -> data, e) == a_g_b)
        {//���ptr.p�ĺ��Ԫ�ش��ڲ���ֵ�����غ��Ԫ�ص�ǰ��ָ�� 
            ptr.type = prior;
            return ptr;
        }
        if(compare( ptr.p -> next -> data, e) == a_s_b)
        {//���ptr.p�ĺ��Ԫ��С�ڲ���ֵ����������� 
            ptr.p = ptr.p -> next;
        }
    }
    ptr.type = nextor;//e�Ƕ���ʽ������Ԫ�� 
    return ptr;
}

//����polyn_print������ӡ����ʽ 
void polyn_print(polynominal HEAD)
{

    polynominal p = HEAD -> next;
    if(!p)
        printf("������Ϊ��\n");
    else
    {
        while(p -> next)
        {
            if (p -> data.coef >= 0){
            	if(p->data.expn == 0) printf("%.3f + ",p -> data.coef);
            	else if(p->data.expn < 0) printf("");
            	else printf("%.3fx^%d + ",p -> data.coef,p -> data.expn);
			} 
                
            else
            	if(p->data.expn == 0) printf("%.3f ",p -> data.coef);
            	else if(p->data.expn < 0) printf("");
                else printf("%.3fx^%d ",p -> data.coef,p -> data.expn);   

            p = p -> next;
        }
 		if(p->data.expn == 0) printf("%.3f\n",p->data.coef);       
		else printf("%.3fx^%d\n",p -> data.coef,p -> data.expn);

    }
    
    
    
}

//����������������ʽ
void polyn_order_insert(polynominal HEAD, datatype e, int(*compare)(datatype, datatype))
{

    locate ptr = polyn_locate(HEAD, e, compare);//�ڲ�����polyn_locate���� 
    if (ptr.type == nextor)
    {
        //β��һ���ڵ� 
        polynominal new_node = (polynominal)malloc(sizeof(Pnode));
        new_node -> data = e;
        ptr.p -> next = new_node;
        new_node -> next = NULL;
    }
    if (ptr.type == prior)
    {
        //��ǰ��֮�����ڵ� 
        polynominal new_node = (polynominal)malloc(sizeof(Pnode));
        new_node -> data = e;
        new_node -> next = ptr.p -> next;
        ptr.p -> next = new_node;
    }
    if (ptr.type == curor)
    {	
		//������������ϲ�ϵ�� 
        (ptr.p -> data).coef += e.coef;
    }
}



//�������ʽ��ʼ������
 void polyn_create(polynominal HEAD, int m)
{
    //ʵϵ��ָ������ʽ 
    int i = 0;
    HEAD -> next = NULL;
    datatype temp;
    scanf("%f %d",&(temp.coef), &(temp.expn));
    while(temp.coef != 0 && i < m)//��Ч����Ϊϵ������
    {
		//�����ʽ�в����µ���ڵ�
        polyn_order_insert(HEAD, temp, compare);
        ++i;
        if(i<m) scanf("%f %d",&(temp.coef), &(temp.expn));
    }
}


//�������ʽ�������� 
void polyn_destroy(polynominal HEAD)
{
    while(HEAD)
    {
        polynominal p = HEAD;
        HEAD = HEAD -> next;
        free(p);
    }
}

//�������ʽ���� 
int polyn_length(polynominal HEAD)
{
    polynominal p = HEAD -> next;
    int i = 0;
    while(p)
    {
        i += 1;
        p = p -> next;
    }
    return i;

}

//��������ʽ��� 
polynominal polyn_add(polynominal pa, polynominal pb)
{ 
    polynominal hc = pa;//������ͷ��� 
    polynominal pc = hc;//�������α� 
    polynominal hb = pb;//����b��ͷ��� 
    pb = pb -> next;//����a��Ԫ��� 
    pa = pa -> next;//����b��Ԫ��� 
    int type;
    while(pa && pb)
    {
        type = compare(pa -> data, pb -> data);
        if (type == a_e_b)
        {
            (pa -> data).coef = (pa -> data).coef + (pb -> data).coef;//ָ����ͬ�ϲ�ϵ�� 
            if (pa -> data.coef == 0)
            {
                pc -> next = pa;//ɾ������pa��� 
                pa = pa -> next;
                free(pc -> next);
                pc -> next = NULL;
                hb -> next = pb -> next;//ɾ������pb��� 
                free(pb);
                pb = hb -> next;

            }
            else
            {
                pc -> next = pa; //������������
                pc = pa;
                hb -> next  = pb -> next;//�ı�b��ͷ��� 
                free(pb);
                pb = hb -> next; 
                pa = pa -> next;
            }
        }
        if (type == a_s_b)
        {
            pc -> next = pa;//������������
            pc = pa;
            pa = pa -> next;
        }
        if (type == a_g_b)
        {
            pc -> next = pb;//������������
            pc = pb;
            pb = pb -> next;
            hb -> next = pb;
        }
    }

    if(pa == NULL)
    {
        if(pb == NULL)
            free(hb);
        else
        {
            pc -> next = pb;
            free(hb);
        }
    }
    else//a�ǿգ�b�� 
    {
        free(hb);
        pc -> next = pa;
    }
    return hc;

}


polynominal polyn_subtract(polynominal pa, polynominal pb)
{
    //�Ȱ�pb��(����)ȡ��,Ȼ����üӷ���������
    //����Ԫ��ʼ�����Ǻ���,��ÿһ��ϵ��ȡ�෴��
    if(!pb || !pa) printf("False");
	polynominal hb = pb; 
    while(hb)
    {
        hb -> data.coef = (0 - (hb -> data).coef);
        hb = hb -> next; 
    }
    polynominal pc = polyn_add(pa, pb);
    return pc;
}

//����ʽ�˷����������ռ䴢������pa��ʼֵ 
polynominal polyn_clone(polynominal HEAD)
{
 
    polynominal pnode  = HEAD;
    polynominal pclone_head,pclone_node;
    if(pnode != NULL)
    {
        pclone_head = (polynominal)malloc(sizeof(Pnode));//��ʼ��clone����ͷָ�� 
		pclone_head->data = pnode->data;//ͷ���ǿ��򿽱����� 
		pclone_head->next = NULL;//����clone����ʽ��ͷ���ָ����Ϊ�� 
		pclone_node = pclone_head;// ����clone����ʽ���α겢����ͷ��� 
		pnode = pnode->next;// ԭ�����α��������Ԫ 
    }
    while(pnode != NULL)
    {
        polynominal temp_node = (polynominal)malloc(sizeof(Pnode));//��ʼ��ת���� 
		temp_node->data = pnode->data;//��pnode�����ݴ���temp_node 
		temp_node->next = NULL;//����ת����ָ����Ϊ�� 
 		pclone_node -> next = temp_node;//��temp_node����pclone������ 
        pclone_node = pclone_node -> next;//�α������temp��� 
        pnode = pnode -> next;//�α���� 
    }
    return pclone_head;
}

//����ʽ�˷� 
polynominal polyn_multiply(polynominal pa, polynominal pb)
{
    polynominal p = (polynominal)malloc(sizeof(Pnode));//�˻�����ռ� 
	p->next = NULL;//����ָ����Ϊ�� 
	polynominal pac = pa->next;//pacָ������pa��Ԫ 
	polynominal pbc = pb->next;//pbcָ������pb��Ԫ 
    while(pbc)
    {
        polynominal pc = polyn_clone(pa);//��������pa����pb������� 
		polynominal pcc = pc->next;//pccָ��pa_clone��Ԫ 
        while(pac)
        {
            pcc->data.coef = (pac->data.coef) * (pbc->data.coef);//pbc����дΪpb���������ڴ���� 
			pcc->data.expn = (pac->data.expn) + (pbc->data.expn);
			pcc = pcc->next;
			pac = pac->next;
		} //pc�д���pa������pb��һ����˵õ������ݶ���ʽ 
		pac = pa->next;//pacָ��pa��Ԫ 
		p = polyn_add(p,pc);//�ϲ�p��pc 
		pbc = pbc->next;//pbcָ������pb�ĵڶ���
    }
    polyn_destroy(pa);
    polyn_destroy(pb);
    return p;
}

//����ʽ��
polynominal polyn_derivate(polynominal pa, int degree){
	if(!pa) printf("Invaid Operation");
	else{
		polynominal ph = pa;
		for(int i = 0;i < degree;i++){
		
		polynominal p = ph->next;//�α�ָ��������Ԫ
		while(p) {
			p->data.coef = (p->data.coef)*(p->data.expn);
			p->data.expn = p->data.expn-1;
			p = p->next;
		}
		}
	return ph;
	}
}

//����ʽ���� 
//-----------------------------------------------------------------------------------------------------------------------------------------
#define STACK_INIT_SIZE 100//�洢�ռ��ʼ����
#define STACKINCREMENT 10//�洢�ռ��������
#define LENGTH 100//����ռ��ʼ������ 

//������ջ�Ľṹ�� 
typedef struct{
	char *base;
	char *top;
	int stacksize;
}OPTR_stack;

//������ջ�Ľṹ��
typedef struct{
	double *base;
	double *top;
	int stacksize;
}OPND_stack;


//������ջ�ĳ�ʼ�� 
void Init_OPTR_stack(OPTR_stack &s){
	s.base = (char*)malloc(STACK_INIT_SIZE * sizeof(char));
	if(!s.base) printf("OPTR_stack Initiation Failed\n");
	s.top = s.base;
	s.stacksize = STACK_INIT_SIZE;
}

//������ջ�ĳ�ʼ�� 
void Init_OPND_stack(OPND_stack &s){
	s.base = (double*)malloc(STACK_INIT_SIZE * sizeof(double));
	if(!s.base) printf("OPND_stack Initiation Failed\n");
	s.top = s.base;
	s.stacksize = STACK_INIT_SIZE;
}

//������ջ��ջ 
void OPTR_stack_Push(OPTR_stack &s, char elem){
	if(s.top - s.base >= s.stacksize){
		s.base = (char*)realloc(s.base, STACKINCREMENT * sizeof(char));
		if(!s.base) printf("OPTR_stack Increment Failed\n");
		
		s.top = s.base + s.stacksize;
		s.stacksize += STACKINCREMENT;	
	}
	*s.top++ = elem;
}

//������ջ��ջ
void OPND_stack_Push(OPND_stack &s, double elem){
	if(s.top - s.base >= s.stacksize){
		s.base = (double*)realloc(s.base, STACKINCREMENT * sizeof(double));
		if(!s.base) printf("OPND_stack Increment Failed\n");
	
		s.top = s.base + s.stacksize;
		s.stacksize += STACKINCREMENT;
	}
	*s.top++ = elem;
} 

//������ջ��ջ 
void OPTR_stack_Pop(OPTR_stack &s, char &elem){
	//if(s.base == s.top) printf("OPTR_stack Empty\n");
	elem = *(--s.top);
} 

//������ջ��ջ 
void OPND_stack_Pop(OPND_stack &s, double &elem){
	//if(s.base == s.top) printf("OPND_stack Empty\n");//
	elem = *(--s.top);
}

//ȡ������ջջ��Ԫ�� 
char OPTR_stack_Top(OPTR_stack &s){
	//if(s.base == s.top) printf("OPTR_stack Empty\n");
	return *(s.top - 1);
} 

// ȡ������ջջ��Ԫ��
double OPND_stack_Top(OPND_stack &s){
	//if(s.base == s.top) printf("OPND_stack Empty\n");
	return *(s.top - 1);
}

//���㺯��2
double Operate(double operand1, char _operator, double operand2){
	switch(_operator){
		case'+':return operand1 + operand2;break;
		case'-':return operand1 - operand2;break;
		case'*':return operand1 * operand2;break;
		case'/':return operand1 / operand2;break;
		default:return 0;
	}
} 
 
//�������жϺ��� 
bool Ischar(char ch){
	if(ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '(' || ch == ')' || ch == '#')
	return 1;
	else return 0;
} 

bool Isnum(char num){
	if(num >= '0' && num <= '9') return 1;
	else return 0;
} 

//��������ȼ��� 
unsigned char prior_table[][7] ={
                {'>', '>', '<', '<', '<', '>', '>'},
                {'>', '>', '<', '<', '<', '>', '>'},
                {'>', '>', '>', '>', '<', '>', '>'},
                {'>', '>', '>', '>', '<', '>', '>'},
                {'<', '<', '<', '<', '<', '=', ' '},
                {'>', '>', '>', '>', ' ', '>', '>'},
                {'<', '<', '<', '<', '<', ' ', '='}
               };

//���ȼ����� 
char Precede(char optr1, char optr2){
	int row;
	int col;
		
	switch(optr1){
		case '+':row = 0;break;
		case '-':row = 1;break;
		case '*':row = 2;break;
		case '/':row = 3;break;
		case '(':row = 4;break;
		case ')':row = 5;break;
		case '#':row = 6;break;
	}
	
	switch(optr2){
		case '+':col = 0;break;
		case '-':col = 1;break;
		case '*':col = 2;break;
		case '/':col = 3;break;
		case '(':col = 4;break;
		case ')':col = 5;break;
		case '#':col = 6;break;		
	}
	
	return prior_table[row][col];
}  

//�ַ����ṹ�� 
typedef struct string{
	char value[100000];
	int length;
}string;

//�ṹ���ʼ�� 
void Init_string(string &s){
	gets(s.value);
	s.length = strlen(s.value);	
} 

//����next���� 
void GetNext(string T, int next[]){
	int i,j;
	i = 0;//ָ���ַ���ÿ���ַ����±�
	j = -1;
	next[i] = j;//��next[0]��Ϊ0
	while(i < T.length){
		if(j == -1 || T.value[i] == T.value[j])
		{++i; ++j; next[i] = j;}
		else j = next[j];
	} 

}

//kmp����,����������ģʽ�������±� 
int KMP(string S, string T, int next[]){//����������ģʽ����next����
	int i,j;
	i = j = 0;
	while(i < S.length && j < T.length){//iΪ�����±꣬jΪģʽ���±� 
		if(j == -1 || S.value[i] == T.value[j])
		{++i; ++j;}
		else j = next[j];
	}
	if(j == T.length) return i - T.length;
	else return -1;
} 


//�ַ�����������(�������ַ����еı������滻Ϊ������ֵ)
void VariableExchange(string variable, string &expression, string key, int index){//��expression�ַ��������ҵ�variable���飬���滻Ϊkey����  
	if(variable.length == key.length){//�������ͱ���ֵ�ȳ�ʱ 
		for(int i = 0;i < variable.length;i++){
			expression.value[i + index] = key.value[i];
		}
	}
	
	if(variable.length < key.length){//��������ֵ�̣���Ҫ��������֮��Ĳ��ֺ���
		expression.length = expression.length + key.length - variable.length;
		
		for(int i = expression.length;i >= index + variable.length;i--){
			expression.value[i + (key.length - variable.length)] = expression.value[i];
		}
		for(int j = 0;j < key.length;j++){//��� 
			expression.value[j + index] = key.value[j];
		} 
	}
	
	if(variable.length > key.length){//��������ֵ������Ҫ��������֮��Ĳ���ǰ��
		
		for(int i = index + variable.length;i < expression.length;i++){
			expression.value[i - (variable.length - key.length)] = expression.value[i];
		}
		for(int k = expression.length + key.length - variable.length;k < expression.length;k++) expression.value[k] = '\0';
		
		expression.length = expression.length + key.length - variable.length;
		
		
		
		for(int j = 0;j < key.length;j++){//��� 
			expression.value[j + index] = key.value[j];
		} 
	}
	

} 

void MonoVariableSwap(string &expression, string variable, string key){
	int *next = (int *)malloc(variable.length * sizeof(int));
	int index;
	GetNext(variable,next);
	index = KMP(expression,variable,next);
	VariableExchange(variable,expression,key,index);
}

//ϸ�ڴ���
void Detail(string &expression){
	string special;
	special.length = 1;
	special.value[0] = '-';
	
	string key;
	key.length = 2;
	key.value[0] = '0';
	key.value[1] = '-';
	MonoVariableSwap(expression,special,key);
} 

void EvalueExpression(char* expression){
	char Operator;
	int i = 0, precision = -1, head = 0;
	double combine, operand1, operand2, string_num;
	
	OPTR_stack OPTR;
	Init_OPTR_stack(OPTR);//��ʼ��������ջ 
	
	OPND_stack OPND;
	Init_OPND_stack(OPND);//��ʼ��������ջ 
	
	OPTR_stack_Push(OPTR,'#');//�������ջ��ѹ��'#' 

	while(expression[i] != '#' || OPTR_stack_Top(OPTR) != '#'){ 
		if(Isnum(expression[i])){//�����ǰ�ַ������� 
			string_num = 0;//���յ���ջԪ��
			while(Isnum(expression[i])||expression[i] == '.'){//δ����whileѭ�� 
				if(expression[i] == '.')
				head = 1;//ʶ��С������ʼ��¼���� 
				else string_num = string_num * 10 + expression[i] - '0';//����ֵ�ַ�ת��Ϊ��Ӧ����
				
				if(head) precision++;//С��λ����һ 
				i++;//�α����						
			}//��ʱ��expression[i]�������� 
			if(precision != -1) string_num = string_num / pow(10,precision);//���ݼ�¼�ľ��Ƚ�������Ϊ������ 
			//printf("��ջ������OPND:%lf\n",string_num);
			OPND_stack_Push(OPND, string_num);
			precision = -1;head = 0;//�ָ����Ⱥ�ʶ���Ĭ��ֵ 
			}
		else{//�����ǰ�ַ��������� 
			switch(Precede(OPTR_stack_Top(OPTR), expression[i])){
				case'<'://printf("��ջ��������%c\n",expression[i]);
						OPTR_stack_Push(OPTR,expression[i]);i++;break;
				
				case'>':
						OPTR_stack_Pop(OPTR,Operator);
						//printf("��ջ��������%c\n", Operator);
						OPND_stack_Pop(OPND,operand1);
						//printf("��ջ������2��%lf\n",operand1);	
						OPND_stack_Pop(OPND,operand2);
						//printf("��ջ������2��%lf\n",operand2);
						combine = Operate(operand2, Operator, operand1);
						//printf("��ջ������:%lf\n",combine);
						OPND_stack_Push(OPND, combine); 
						break;
				
				case'=':OPTR_stack_Pop(OPTR,Operator);
						i++;
						break; 
			} 
		}
	}
	printf("%.3lf\n",OPND_stack_Top(OPND));	
}

//--------------------------------------------------------------------------------------------------------------------------






int main()
{	
	printf("��ѡ����㹦�ܣ�\n");
	printf("����1������������ģʽ\n");
	printf("����2�������ʽ����ģʽ��˳�����\n");
	printf("����3�������ʽ����ģʽ����ʽ����\n");
	printf("����4�������ʽ��ֵ����ģʽ\n");
	printf("����5���뵥��������ʽ��ֵ����ģʽ\n"); 
	printf("����6�˳�������\n");
	printf("�������㹦�ܣ�+ - ^(�н�����ֵ)\n");
	printf("����ʽ���㹦�ܣ�+ - * d(��)\n"); 
	 
	
	loop:int mode;
	printf("---------------------------------------------------------\n");
	scanf("%d",&mode);
	
	if(mode == 1){
	//----------------------------------------��ʼ������------------------------------//
 	int dim1;
	int dim2;

	printf("***************����+���������ĺ�*****************\n");
	printf("***************����-���������Ĳ�*****************\n");
	printf("***************����^�������ʽ�ļн�����ֵ***************\n");	 
	printf("��������1ά�ȣ�");
	scanf("%d",&dim1);
	 
	Vector va;
	Vector vb;
	
	Init_Vector(va,dim1);
	Print_Vector(va);
	
	char _operator;
    
	if(getchar() == '\n'){
		printf("�����������");
    	scanf("%c",&_operator);
	}
	
	if(_operator == '+'){
		printf("��������2ά�ȣ�");
		scanf("%d",&dim2);
		Init_Vector(vb,dim2);
		Print_Vector(vb);
		
		printf("��������\n");
		vector result = Vector_add(&va,&vb);
		Print_Vector(*result); 
	}

	if(_operator == '-'){
		printf("��������2ά�ȣ�");
		scanf("%d",&dim2);
		Init_Vector(vb,dim2);
		Print_Vector(vb);
		
		printf("��������\n");
		vector result = Vector_subtract(&va,&vb);
		Print_Vector(*result);
	}


	if(_operator == '^'){
		printf("��������2ά�ȣ�");
		scanf("%d",&dim2);
		Init_Vector(vb,dim2);
		Print_Vector(vb);
		
		double cos = Vector_cos(&va,&vb);
		printf("��������\n");
		printf("%lf\n",cos); 
	}
	goto loop;
	
	}
	
	if(mode == 2){
	//����ʽ����˳���ʵ��
	int terms1;
	int terms2;
	printf("***************����+�������ʽ�ĺ�*****************\n");
	printf("***************����-�������ʽ�Ĳ�*****************\n");
	printf("***************����*�������ʽ�Ļ�*****************\n");
	printf("***************����d�������ʽ�ĵ���***************\n");	
	 
	printf("�������ʽ1������") ;
	scanf("%d",&terms1);	
	
	ArrayPoly a1;
	Init_ArrayPoly(a1,terms1);//��ʼ������ʽa1
	Print_ArrayPoly(a1);
	
	char _operator;
	if(getchar() == '\n'){
		printf("�����������");
    	scanf("%c",&_operator);
	} 
	
		if(_operator == '+'){
			
		printf("�������ʽ2������") ;
		scanf("%d",&terms2);
		
		ArrayPoly a2;
		Init_ArrayPoly(a2,terms2);//��ʼ������ʽa1
		Print_ArrayPoly(a2);
		
		Polyptr result = Add_ArrayPoly(&a1,&a2);
		printf("��������\n");
		Print_ArrayPoly(*result); 
	
	goto loop;		
	}
	
	if(_operator == '-'){
			
		printf("�������ʽ2������") ;
		scanf("%d",&terms2);
		
		ArrayPoly a2;
		Init_ArrayPoly(a2,terms2);//��ʼ������ʽa1
		Print_ArrayPoly(a2);

		Polyptr result = Subtract_ArrayPoly(&a1,&a2);
		printf("��������\n");
		Print_ArrayPoly(*result); 	
    goto loop;		
	}
	
	if(_operator == '*'){
			
		printf("�������ʽ2������") ;
		scanf("%d",&terms2);
		
		ArrayPoly a2;
		Init_ArrayPoly(a2,terms2);//��ʼ������ʽa1
		Print_ArrayPoly(a2);
		
		Polyptr result = Multiply_ArrayPoly(&a1,&a2);
		printf("��������\n");
		Print_ArrayPoly(*result);
		 
	goto loop;
	}
	
	if(_operator == 'd'){
			
		int degree;
		printf("�����󵼽�����") ;
		scanf("%d",&degree);
		
		Polyptr result = Diff_ArrayPoly(&a1,degree);
		Print_ArrayPoly(*result);
		
	goto loop; 
	}
	}
	
	if(mode == 3){
	//����ʽ������ʽ��ʵ�� 
	//----------------------------------------��ʼ������ʽ----------------------------//
	int length1;
	int length2;
	
	printf("***************����+�������ʽ�ĺ�*****************\n");
	printf("***************����-�������ʽ�Ĳ�*****************\n");
	printf("***************����*�������ʽ�Ļ�*****************\n");
	printf("***************����d�������ʽ�ĵ���***************\n");
	
	printf("�������ʽ1������") ;
	scanf("%d",&length1);
	 
    polynominal pa = (polynominal)malloc(sizeof(Pnode));
    polyn_create(pa,length1);

    polyn_print(pa);
	putchar('\n') ;
	
	char _operator;
    
	if(getchar() == '\n'){
		printf("�����������");
    	scanf("%c",&_operator);
	} 
//----------------------------------------����ʽ��������----------------------------// 

		
	if(_operator == '+'){
			
		printf("�������ʽ2������") ;
		scanf("%d",&length2);
	
    	polynominal pb = (polynominal)malloc(sizeof(Pnode));
    	polyn_create(pb,length2);
    	polyn_print(pb);
    	putchar('\n') ;
		
		polynominal p_add =  polyn_add(pa, pb);
	    //putchar('\n') ;
		polyn_print(p_add);
	goto loop;
	}

		
	if(_operator == '-'){
			
		printf("�������ʽ2������") ;
		scanf("%d",&length2);
		
    	polynominal pb = (polynominal)malloc(sizeof(Pnode));
    	polyn_create(pb,length2);
    	polyn_print(pb);
    	putchar('\n') ;
		
		polynominal p_sub =  polyn_subtract(pa, pb);
		//putchar('\n') ;
		polyn_print(p_sub);
	goto loop;	
	}

		
	if(_operator == '*'){
			
		printf("�������ʽ2������") ;
		scanf("%d",&length2);
	
    	polynominal pb = (polynominal)malloc(sizeof(Pnode));
    	polyn_create(pb,length2);
    	polyn_print(pb);
    	putchar('\n') ;
		
		polynominal p_multi = polyn_multiply(pa, pb);
		//putchar('\n') ;
    	polyn_print(p_multi);
    goto loop;
	}
	
	
	if(_operator == 'd'){
		
		int degree;
		printf("�����󵼽�����") ;
		scanf("%d",&degree);
		
		polynominal p_d1 = polyn_derivate(pa,degree);
		polyn_print(p_d1);
	goto loop;
	}
	} 
	
	if(mode == 4){
	getchar(); 
	printf("�������ʽ��\n"); 
	string expression;
	Init_string(expression);
	Detail(expression);

	EvalueExpression(expression.value);
	goto loop;
	} 
	
	if(mode == 5){
	getchar();
	
	int count = 0;//��¼�������� 
	printf("�����������\n");
	
	string variable;
	Init_string(variable);
	
	printf("���뵥����������\n"); 
	scanf("%d",&count);	
	getchar(); 	
	
	printf("�������ʽ��\n"); 
	string expression;
	Init_string(expression);
	
	printf("������ֵΪ��\n");
	string key;
	Init_string(key);

	for(int j = 0;j < count;j++){
	Detail(expression);
	MonoVariableSwap(expression,variable,key);
	}
	
	puts(expression.value);
	
	
	puts(expression.value);

	EvalueExpression(expression.value); 
	goto loop;
	}
	
	if(mode == 6){//�˳������� 
	return 0;
	} 
}



