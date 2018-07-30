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
