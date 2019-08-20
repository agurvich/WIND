#include <stdio.h>

extern struct Point{
    int x;
    int y;
} Point;

__global__ void readStruct(
    struct Point point
    ){
    printf("x: %d\n",point.x);
    printf("y: %d\n",point.y);
}

int main(){

    struct Point h_point = {0};
    h_point.x = 1;
    h_point.y = 2;

    readStruct<<<1,1>>>(h_point);
    
}
