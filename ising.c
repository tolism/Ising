//! Ising model evolution
/*!

  \param G      Spins on the square lattice             [n-by-n]
  \param w      Weight matrix                           [5-by-5]
  \param k      Number of iterations                    [scalar]
  \param n      Number of lattice points per dim        [scalar]

  NOTE: Both matrices G and w are stored in row-major format.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define old(i,j,n) *(old+i*n+j)
#define new(i,j,n) *(new+i*n+j)
#define w(i,j) *(w+i*5+j)
#define G(i,j,n) *(G+i*n+j)

void swapElement(int  ** one, int  ** two) {
  int  * temp = * one;
  * one = * two;
  * two = temp;
}

double influenceCalc(int *old , double *w , int n , int i , int j  ){
  double influence = 0;
  for(int ii=0; ii<5; ii++){
    for(int jj=0; jj<5; jj++){
      influence +=  w(ii,jj) * old((i-2+n+ii)%n, (j-2+n+jj)%n, n);
    }
  }
return influence;
}

void ising( int *G, double *w, int k, int n){

  int * old = (int*) malloc(n*n*sizeof(int)); // old spin lattice
  int * new = (int*) malloc(n*n*sizeof(int)); // new spin lattice
//  int * tmp;
  double influence; // weighted influence of the neighbors

//Elearning tester checks the values of the G so by swaping
// The "head" pointer it can not pass the validation
// So we manual copy
 //swapElement(&old,&G);
  //initial values saved to old
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
        old(i,j,n) = G(i,j,n);
    }
  }

  // run for k steps
  for(int l=0; l<k; l++){

    // for each new[i][j] point
    for(int i=0; i<n; i++){
      for(int j=0; j<n; j++){

        // calculation of weighted influence
        influence = influenceCalc(old , w ,n ,i , j);

        // magnetic moment gets the value of the SIGN of the weighted influence of its neighbors
        if(fabs(influence) < 10e-7){
          new(i,j,n) = old(i,j,n); // remains the same in the case that the weighted influence is zero
        }
        else if(influence > 10e-7){
          new(i,j,n) = 1;
        }
        else if(influence < 0){
          new(i,j,n) = -1;
        }
        // save result in G
        G(i,j,n) = new(i,j,n);
      }
    }

    // swap the pointers for the next iteration
    swapElement(&old,&new);
    // tmp = old;
    // old = new;
    // new = tmp;

    // terminate if no changes are made
    int areEqual = 0;
    for(int i=0; i<n; i++){
      for(int j=0; j<n; j++){
        if(old(i,j,n) == new(i,j,n)){
          areEqual++;
        }
      }
    }
    // termination branch
    if(areEqual == n*n){
      printf("terminated: spin values stay same (step %d)\n" , l);
      exit(0);
    }
  }
}

int main(int argc, const char* argv[]){

  int n = 517;
  int k = 11;
  double weights[] = {0.004, 0.016, 0.026, 0.016, 0.004, 0.016, 0.071, 0.117, 0.071, 0.016, 0.026, 0.117, 0.0, 0.117, 0.026, 0.016, 0.071, 0.117, 0.071, 0.016, 0.004, 0.016, 0.026, 0.016, 0.004};
  int G[n*n]; // G that changes k times
  int Gk[n*n]; // file from floros-albania a.e.
  FILE *ptr;

  // read initial G
  ptr = fopen("conf-init.bin","rb");
  fread(G,sizeof(G),1,ptr);
  fclose(ptr);
  // read k-th Gk
  ptr = fopen("conf-11.bin","rb"); // allazo onoma arxeiou gia allagi k
  fread(Gk,sizeof(Gk),1,ptr);
  fclose(ptr);

  // execution
  ising(G, weights, k, n);

  // check correctness
  int c = 0;
  for(int i=0; i<517; i++){
    for(int j=0; j<517; j++){
      if( *(G+i*517+j) != *(Gk+i*517+j) ){
        c++;
      }
    }
  }
  if(c!=0){
    printf("malakia\n");
  }
  else{
    printf("success\n");
  }

}
