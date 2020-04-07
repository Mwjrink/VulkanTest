
// THIS IS LITERALLY because I dont like seeing raw pointers anywhere so
namespace std
{
    template <typename T>
    using observer_ptr = T*;
}