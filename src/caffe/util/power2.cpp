// IAO具体量化方法=======================
#include "caffe/util/power2.hpp"
#include "caffe/common.hpp"
#include <math.h>

namespace caffe
{
// 量化 权重值----->  pow(2,i)=========
// weight 量化为  +pow(2,ni) / -pow(2,ni) , ni: n1-7:n1, 选择量化误差最小的一个
  template <typename Dtype>
  double weightCluster( Dtype weight, int M)
  {
    double min=100;
    double ind=0;
    double flag=1.0;
	
	// 最小值
    if(min > std::abs(weight))
    {
      min = std::abs(weight);
      flag=0.0;// 权重绝对值未超过100
    }
	  
// the number 7 (default, n2的值 ) is corresponding to 5 bits in paper,
//  you can modify it, 3 for 4 bits, 1 for 3 bits, 0 for 2 bits.
// n2 = n1 + 1 −2^(b−1)/2. 
// For instance, if b = 3(量化精度) and n1 = −1, it is easy to get n2 = −2, 
// if b=5, n1 = −1， n2= -1 + 1-(2^(5-1))/2 = -8
// 0 for 2 bits
// 1 for 3 bits
// 3 for 4 bits
// 7 for 5 bits
// 15 for 6 bits
// 31 for 7 bits     2^b - 1
// 63 for 8 bits    
    for(int i=(M-7); i<=M; i++)
      {
		  
        if(min > std::abs(weight - pow(2,i)))//    weight 量化为  +pow(2,i) 的  量化差值
          {
            min = std::abs(weight - pow(2,i));//     最小量化差值
            ind=i;
            flag=1.0;// weight 量化为  +pow(2,i)
          }
		  
        if(min > std::abs(weight + pow(2,i)))//   weight 量化为  -pow(2,i) 的 量化差值
          {
            min = std::abs(weight + pow(2,i));//    最小量化差值
            ind = i;
            flag = -1.0;
          }
      }
      return flag*pow(2,ind);
  }
  
 template <typename Dtype>
  double weightCluster_zero( Dtype weight, int M)
  {
    double min=100;
    double ind=0;
    double flag=1.0;
    if(min > std::abs(weight))
    {
      min=std::abs(weight);
      flag=0.0;
    }
	 
// the number 7 (default, n2的值 ) is corresponding to 5 bits in paper,
//  you can modify it, 3 for 4 bits, 1 for 3 bits, 0 for 2 bits.
// n2 = n1 + 1 −2^(b−1)/2. 
// For instance, if b = 3(量化精度) and n1 = −1, it is easy to get n2 = −2, 
// if b=5, n1 = −1， n2= -1 + 1-(2^(5-1))/2 = -8
// 0 for 2 bits
// 1 for 3 bits
// 3 for 4 bits
// 7 for 5 bits
// 15 for 6 bits
// 31 for 7 bits     2^b - 1
// 63 for 8 bits
    for(int i=(M-7); i<=M; i++)// 从最高比特位 M=n1 进行遍历
      {
        if(min > std::abs(weight - pow(2,i)))
          {
            min = std::abs(weight - pow(2,i));
            ind=i;
            flag=1.0;
          }
        if(min > std::abs(weight + pow(2,i)))
          {
            min=std::abs(weight + pow(2,i));
            ind = i;
            flag = -1.0;
          }
      }
      return flag*pow(2,ind);
  }

  template double weightCluster<float>(float weight,int M);
  template double weightCluster<double>(double weight,int M);
  template double weightCluster<unsigned int>(unsigned int weight,int M);
  template double weightCluster<int>(int weight,int M);
  
  template double weightCluster_zero<float>(float weight,int M);
  template double weightCluster_zero<double>(double weight,int M);
  template double weightCluster_zero<unsigned int>(unsigned int weight,int M);
  template double weightCluster_zero<int>(int weight,int M);
}
