#ifndef HeterogeneousCore_MPICore_MPIToken_h
#define HeterogeneousCore_MPICore_MPIToken_h

#include <memory>

// forward declaration
class MPISender;

class MPIToken {
public:
  // default constructor, needed to write the type's dictionary
  MPIToken() = default;

  // user-defined constructor
  explicit MPIToken(std::shared_ptr<MPISender> link) : link_(link) {}

  // access the data member
  MPISender* link() const { return link_.get(); }

private:
  // wrap the MPI communicator and destination
  std::shared_ptr<MPISender> link_;
};

#endif  // HeterogeneousCore_MPICore_MPIToken_h
