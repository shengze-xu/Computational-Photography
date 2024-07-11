#include<iostream>
#include<vector>
#include <algorithm>
#include <math.h>
using namespace std;

class Node{
	public:
		int row;
		int col;
		double value;
		Node *col_next;
		Node *row_next;
		Node(int row,int col,double value){
			this->row=row;
			this->col=col;
			this->value=value;
			this->col_next=nullptr;
			this->row_next=nullptr;
		} 
		Node(){
			this->row=0;
			this->col=0;
			this->value=0;
			this->col_next=nullptr;
			this->row_next=nullptr;
		}
};

class Sparsematrix{
	public:
		int rows;
		int cols;
		int number;
		Node *element;
		
		Sparsematrix(int row_number,int col_number):rows(row_number),cols(col_number),number(0){
			element=new Node[rows];
		}	
		
		double at(int row,int col){
			Node *row_now=&element[row];
			Node *ptr=nullptr;
			while(row_now){
				if(row_now->col==col){
					ptr=row_now;
				}
				row_now=row_now->col_next;
			}
			if(ptr==nullptr){
				return 0;
			}
			else{
				return ptr->value;
			}
		}
		
		Node *insert(double val,int row,int col){
			if(val==0) return nullptr;
			Node *row_now=&element[row];
			while(row_now){
				if(row_now->col==col){
					if(row_now->value==0){
						number++;
					}
					row_now->value=val;
					return row_now;
				}
				if(row_now->col<col && row_now->col_next && row_now->col_next->col>col){
					Node *nele=new Node(row,col,val);
					number++;
					nele->col_next=row_now->col_next;
					row_now->col_next=nele;
					return nele;
				}
				if(row_now->col_next==nullptr){
					Node *nele=new Node(row,col,val);
					number++;
					row_now->col_next=nele;
					return nele;
				}
				row_now=row_now->col_next;
			}
		}
		
		void initializeFromVector(vector<double> rows, vector<double> cols, vector<double> vals){
			delete this->element;
			this->number=0;
			this->rows=*max_element(rows.begin(), rows.end()) + 1;
			this->cols=*max_element(cols.begin(), cols.end()) + 1;
			element = new Node[this->rows];;
			for(int i=0;i<rows.size();i++){
				insert(vals[i],rows[i],cols[i]);
			}
		}
		
		void printmatrix() {
			cout << "matrix " << this->rows << "¡Á" << this->cols << ":" << endl;
			for (int i = 0; i < this->rows; i++) {
				for (int j = 0; j < this->cols; j++) {
					cout << this->at(i, j) << "  ";
				}
				cout << endl;
			}
		}
		 
		static bool limit(vector<double>& v1, vector<double>& v2,double error) {
			double res = 0.0;
			for (int i = 0; i < v1.size(); i++) {
				res += abs(v1[i] - v2[i]);
			}
			if (res <=error)
				return true;
			else
				return false;
		}
		
		vector<double> Gauss_Seidel(double B[]){
			vector <double> result;
			vector <double> pre_result;
			result.assign(this->rows,0);
			int num=0;
			double error=0.000001;
			do{
				num++;
				pre_result=result;
				if(num==1) pre_result.assign(this->rows,1.0);
				for(int i=0;i<this->rows;i++){
					if(at(i,i)==0) continue;
					double sum1=0;
					double sum2=0;
					for(int j=0;j<i;j++){
						sum1+=at(i,j)*result[j];
					}
					for(int j=i+1;j<this->rows;j++){
						sum2+=at(i,j)*pre_result[j];
					}
					result[i]=(B[i]-sum1-sum2)/at(i,i);
				}
			}while(!limit(result,pre_result,error));
			return result;
		}
		
		
		vector<double> Ax(vector<double> x){
			vector<double> Ax_k;
			for (int i=0;i<this->rows;i++) {
				double sum=0;
				for (int j=0;j<this->cols;j++) {
					sum+=this->at(i, j)*x[j];
				}
				Ax_k.push_back(sum);
			}
			return Ax_k;
		}
		
		vector<double> Conjugate_gradient(double B[]){
			vector<double> xk;
			//cout<<"1....."<<endl;
			xk.assign(this->rows,0);
			vector<double> b;
			for(int i=0;i<this->rows;i++){
				b.push_back(B[i]);
			}
			vector<double> rk;
			for(int i=0;i<b.size();i++){
				vector<double> Ax_k;
				Ax_k=this->Ax(xk);
				rk.push_back(b[i]-Ax_k[i]);
			}
			vector<double> pk=rk;
			int k=0;
			//cout<<"2....."<<endl;
			while(1){
				double rrk=0;
				for(int i=0;i<rk.size();i++){
					rrk+=rk[i]*rk[i];
				}
				vector<double> Ap_k=this->Ax(pk);
				double pAp=0;
				for(int i=0;i<pk.size();i++){
					pAp+=pk[i]*Ap_k[i];
				}
				double alpha_k=rrk/pAp;
				vector<double> xk1;
				for(int i=0;i<xk.size();i++){
					xk1.push_back(xk[i]+alpha_k*pk[i]);
				}
				xk=xk1;
				vector<double> rk1;
				double err=0;
				for(int i=0;i<rk.size();i++){
					rk1.push_back(rk[i]-alpha_k*(this->Ax(pk)[i]));
					err+=abs(rk1[i]);
				}
				if(err<0.00000001) break;
				double rrk1=0;
				for(int i=0;i<rk1.size();i++){
					rrk1+=rk1[i]*rk1[i];
				}
				double beta_k=rrk1/rrk;
				vector<double> pk1;
				for(int i=0;i<rk1.size();i++){
					pk1.push_back(rk1[i]+beta_k*pk[i]);
				}				
				pk=pk1;
				rk=rk1;
				k++;
			}
			return xk;
		}
};


int main(){
	//²âÊÔinsert()º¯ÊýºÍat()º¯Êý 
	Sparsematrix s(3 ,3);
	s.insert(6, 0, 0);
	s.insert(7, 0, 1);
	s.insert(5, 1, 2);
	s.insert(4, 2, 1);
	cout << s.at(0, 0) << endl;
	cout << s.at(0, 1) << endl;
	cout << s.at(1, 0) << endl;
	cout << s.at(2, 1) << endl;
	s.printmatrix();
	
	//²âÊÔinitializeFromVector()º¯ÊýºÍGauss_Seidel()º¯Êý 
	vector<double> r1 = { 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3 };
	vector<double> c1 = { 0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3 };
	vector<double> v1 = { 10,-1,2,0,-1,11,-1,3,2,-1,10,-1,0,3,-1,8 };
	s.initializeFromVector(r1, c1, v1);
	s.printmatrix();
	double B[] = { 6,25,-11,15 };
	vector<double> x=s.Gauss_Seidel(B);
	cout << "Solution:" << endl;
	for (int i = 0; i < s.cols; i++)
		cout << x[i] << " ";
	cout << endl;
	
	//²âÊÔConjugate_gradient()º¯Êý 
	vector<double> r2 = { 0,0,1,1 };
	vector<double> c2 = { 0,1,0,1 };
	vector<double> v2 = { 7,1,1,-1 };
	s.initializeFromVector(r2, c2, v2);
	s.printmatrix();
	double B2[] = { 4,-3 };
	vector <double>x2=s.Conjugate_gradient(B2);
	cout << "Solution:" << endl;
	for(int i=0;i<s.rows;i++){
		cout << x2[i] << " ";
	}
	cout << endl;
} 
