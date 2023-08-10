// foo.cpp

#include "pcl/point_types.h"
#include "pcl/impl/instantiate.hpp"
#include "foo.h"
#include "foo.hpp"

// Instantiations of specific point types
PCL_INSTANTIATE(Foo, PCL_XYZ_POINT_TYPES));