#include "hafutils.h"

void dec2bin(char *dst, unsigned long long int x, unsigned char len)
{
  /*
  Convert decimal number in  x to character vector dst of length len 
  representing binary number
  */

  char i; // this variable cannot be unsigned
  for (i = len - 1; i >= 0; --i)
    *dst++ = x >> i & 1;
}


unsigned char find2(char *dst, unsigned char len, unsigned char *pos){
  /* Given a string of length len
     it finds in which positions it has a one
     and stores its position i, as 2*i and 2*i+1 in consecutive slots
     of the array pos.
     It also returns (twice) the number of ones in array dst
  */
  unsigned char sum=0;
  unsigned char j=0;
  for(unsigned char i=0;i<len;i++){
    if(1==dst[i]){
      sum++;
      pos[2*j]=2*i;
      pos[2*j+1]=2*i+1;
      j++;
    }
  }
  return 2*sum;
}
