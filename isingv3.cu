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
#include <time.h>


#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

#define GRID_SIZE_X 517
#define GRID_SIZE_Y 517

#define WSIZE 5
#define RADIUS 2

#define old(i,j,n) *(old+(i)*n+j)
#define current(i,j,n) *(current+(i)*n+j)
#define w(i,j) *(w+(i)*5+j)
#define d_w(i,j) *(d_w+(i)*5+j)
#define s_w(i,j) *(s_w+(i)*5+j)
#define d_current(i,j,n) *(d_current+(i)*n+j)
#define d_old(i,j,n) *(d_old+(i)*n+j)
#define G(i,j,n) *(G+(i)*n+j)
#define s_old(i,j) *(s_old + (i)*(BLOCK_SIZE_X+2*RADIUS) + j)




void swapElement(int  ** one, int  ** two) {
  int  * temp = * one;
  * one = * two;
  * two = temp;
}

__global__
   void kernel2D(int *d_current, int *d_old, double *d_w, int n , int * d_flag){

    double influence = 0;

   int r = blockIdx.x * blockDim.x + threadIdx.x;
   int c = blockIdx.y * blockDim.y + threadIdx.y;

   int lxindex = threadIdx.x + RADIUS;
   int lyindex = threadIdx.y + RADIUS;

   int strideX = blockDim.x*gridDim.x ;
   int strideY = blockDim.y*gridDim.y ;

   __shared__ double s_w[WSIZE*WSIZE];
   __shared__ int s_old[(BLOCK_SIZE_X + 2*RADIUS)*(BLOCK_SIZE_Y + 2*RADIUS)];

   if(blockDim.x > WSIZE && blockDim.y > WSIZE ){
     //Shared weights incase we have sufficient amount of threads ( threads at X > WSIZE AND  threads at Y > WSIZE)
     if(threadIdx.x < WSIZE && threadIdx.y < WSIZE){
       s_w(threadIdx.x, threadIdx.y) = d_w(threadIdx.x, threadIdx.y);
     }
   }
   //threadIdx.x == 0  , threadIdx.y == 0 will do the job
   else{
   if(threadIdx.x == 0 && threadIdx.y == 0){
     for(int i = 0 ; i<WSIZE ; i++){
       for(int j =0; j<WSIZE ; j++){
         s_w(i,j)=d_w(i,j);
       }
     }
   }
  }
  __syncthreads();

   // // Read input elements into shared memory
   for(int i=r; i<n+RADIUS; i+= strideX){
     for(int j=c; j<n+RADIUS ; j+= strideY ){

       // window for old to shared
       s_old(lxindex,lyindex) = d_old((i+n)%n,(j+n)%n,n);
       __syncthreads();

       //stencil kata x
       if( threadIdx.y < RADIUS){
         s_old( lxindex , lyindex - RADIUS ) = d_old((i + n)%n , (j-RADIUS+n)%n , n);
         s_old(lxindex  , lyindex + BLOCK_SIZE_Y ) = d_old( (i + n )%n , (j+BLOCK_SIZE_X + n)%n , n);
       }

       //stencil kata y
       if( threadIdx.x < RADIUS){
         s_old(lxindex  - RADIUS,lyindex ) = d_old( (i-RADIUS+n)%n , (j+n)%n   , n );
         s_old( lxindex + BLOCK_SIZE_Y,lyindex ) = d_old(( i +BLOCK_SIZE_X +n)%n , (j +  n)%n     , n );
       }

       //tetragwnakia stis gwnies
       if(threadIdx.x < RADIUS && threadIdx.y < RADIUS ){
         //panw aristera
         s_old(lxindex - RADIUS , lyindex - RADIUS ) = d_old ( ( i - RADIUS + n)%n , (j-RADIUS + n )%n  , n );
         //katw deksia
         s_old(lxindex +  BLOCK_SIZE_X , lyindex + BLOCK_SIZE_Y  ) = d_old( (i  + BLOCK_SIZE_X + n  )%n , (j + BLOCK_SIZE_Y +n  )%n , n);
         //katw aristera
         s_old(lxindex - RADIUS, lyindex +  BLOCK_SIZE_Y  )= d_old ((i - RADIUS + n)%n  , (j  + BLOCK_SIZE_X )%n, n);
         //panw deksia
         s_old(lxindex + BLOCK_SIZE_X,lyindex - RADIUS  ) = d_old ((i + BLOCK_SIZE_X+n  )%n , (j -RADIUS + n )%n   , n );
       }
       __syncthreads();

       if((i<n)&&(j<n)){
          influence = 0;
           for(int ii=0; ii<WSIZE ; ii++){
             for(int jj=0 ; jj<WSIZE ; jj++){
               if((ii==2) && (jj==2))
                  continue;
            //    influence +=  d_w(ii,jj) * d_old((i-2+n+ii)%n, (j-2+n+jj)%n, n);
              influence += s_w(ii,jj) * s_old( threadIdx.x + ii,threadIdx.y +  jj );
              }
             }

           // // magnetic moment gets the value of the SIGN of the weighted influence of its neighbors
           if(fabs(influence) < 10e-7){
             d_current(i,j,n) =  s_old(lxindex, lyindex  ); // remains the same in the case that the weighted influence is zero
           }
           else if(influence > 10e-7){
             d_current(i,j,n) = 1;
             *d_flag = 0;

           }
           else if(influence < 0){
             d_current(i,j,n) = -1;
             *d_flag = 0;
           }
       }
      __syncthreads();
     }
   }

}


void ising( int *G, double *w, int k, int n){

  dim3 block(BLOCK_SIZE_X,BLOCK_SIZE_Y);
  dim3 grid((GRID_SIZE_X+block.x-1)/block.x,(GRID_SIZE_Y+block.y - 1)/block.y);

      int * old = (int*) malloc(n*n*(size_t)sizeof(int)); // old spin lattice
      int * current = (int*) malloc(n*n*(size_t)sizeof(int)); // current spin lattice

      if(old==NULL || current == NULL){
        printf("Problem at memory allocation at host \n");
        exit(0);
      }

  int * d_old;
  int * d_current;
  double * d_w;// na valw void ** skatakia
  int *d_flag ;
  int flag ;

  if( cudaMalloc((void **)&d_old ,n*n*(size_t)sizeof(int)) != cudaSuccess  || cudaMalloc((void **)&d_current,n*n*(size_t)sizeof(int))   != cudaSuccess   || cudaMalloc((void **)&d_w, WSIZE*WSIZE*(size_t)sizeof(double))   != cudaSuccess || cudaMalloc(&d_flag,(size_t)sizeof(int)) !=cudaSuccess){
    printf("Problem at memory allocation");
    exit(0);
  }

  cudaMemcpy(d_w, w, WSIZE*WSIZE*sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy(d_old, G, n*n*sizeof(int), cudaMemcpyHostToDevice );


  // run for k steps
  for(int l=0; l<k; l++){
    flag = 0;
    kernel2D<<<grid,block>>>(d_current, d_old, d_w, n  , d_flag );
  //  kernel2D<<<dimGrid,dimBlock>>>(d_current, d_old, d_w, n );
    cudaDeviceSynchronize();

  //  cudaMemcpy(old, d_old, n*n*sizeof(int), cudaMemcpyDeviceToHost );
    cudaMemcpy(current, d_current, n*n*sizeof(int), cudaMemcpyDeviceToHost );
    // save result in G
    memcpy(G , current , n*n*sizeof(int));

    // swap the pointers for the next iteration
    swapElement(&d_old,&d_current);

    cudaMemcpy(&flag , d_flag , (size_t)sizeof(int), cudaMemcpyDeviceToHost);
    // terminate if no changes are made
    if(flag){
      printf("No changes \n");
      exit(0);
    }

  }

  free(old);
  free(current);
  cudaFree(d_old);
  cudaFree(d_current);
  cudaFree(d_w);
}




int main(int argc, const char* argv[]){


  int n = 517;
  double weights[] = {0.004, 0.016, 0.026, 0.016, 0.004, 0.016, 0.071, 0.117, 0.071, 0.016, 0.026, 0.117, 0.0, 0.117, 0.026, 0.016, 0.071, 0.117, 0.071, 0.016, 0.004, 0.016, 0.026, 0.016, 0.004};
  int G1[n*n]; // G that changes k times
  int G2[n*n]; // G that changes k times
  int G3[n*n]; // G that changes k times
  int Gk1[n*n];
  int Gk2[n*n];
  int Gk3[n*n];
  FILE *ptr;

  // read initial G
  ptr = fopen("conf-init.bin","rb");
  fread(G1,sizeof(G1),1,ptr);
  fclose(ptr);
  // read initial G
  ptr = fopen("conf-init.bin","rb");
  fread(G2,sizeof(G2),1,ptr);
  fclose(ptr);
  // read initial G
  ptr = fopen("conf-init.bin","rb");
  fread(G3,sizeof(G3),1,ptr);
  fclose(ptr);
  // read k-th Gk
  ptr = fopen("conf-1.bin","rb"); // allazo onoma arxeiou gia allagi k
  fread(Gk1,sizeof(Gk1),1,ptr);
  fclose(ptr);

  ptr = fopen("conf-4.bin","rb"); // allazo onoma arxeiou gia allagi k
  fread(Gk2,sizeof(Gk2),1,ptr);
  fclose(ptr);

  ptr = fopen("conf-11.bin","rb"); // allazo onoma arxeiou gia allagi k
  fread(Gk3,sizeof(Gk3),1,ptr);
  fclose(ptr);

  // execution
  ising(G1, weights, 1, n);

  // check correctness
  int c = 0;
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      if( *(G1+i*n+j) != *(Gk1+i*n+j) ){
        printf("(%d,%d), real:%d, ours:%d\n", i,j, *(Gk1+i*517+j) , *(G1+i*517+j));
        c++;
      }
    }
  }

  if(c!=0){
    printf("k=1 Wrong\n");
  }
  else{
    printf("k=1 Correct\n");
  }

  // execution
  ising(G2, weights, 4, n);

  // check correctness
   c = 0;
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      if( *(G2+i*n+j) != *(Gk2+i*n+j) ){
        c++;
      }
    }
  }
  if(c!=0){
    printf("k=4 Wrong\n");
  }
  else{
    printf("k=4 Correct\n");
  }

  // // execution
  // ising(G, weights, 100, 1000);
  // execution
  ising(G3, weights, 11 , n);

  // check correctness
   c = 0;
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      if( *(G3+i*n+j) != *(Gk3+i*n+j) ){
        c++;
      }
    }
  }
  if(c!=0){
    printf("k=11 Wrong\n");
  }
  else{
    printf("k=11 Correct\n");
  }
  srand(time(NULL));
  int tn = 700;
  int G[tn*tn]; // G that changes k times
  for(int i=0; i<tn; i++){
    for(int j =0; j<tn; j++){

      *(G+i*tn+j) = 2*(rand()%2)-1;

    }
  }
  clock_t start = clock();

  // execution
  ising(G, weights, 5, tn);

  clock_t end = clock();
  double exec_time = (end - start)/(double)CLOCKS_PER_SEC;

  printf("%lf\n", exec_time);


}
