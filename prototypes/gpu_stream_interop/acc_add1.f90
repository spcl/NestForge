! OpenACC Fortran kernel, compiled to libacc_add1.a (mimics a numpyto-openacc ExternalCall variant).
! Device-pointer ABI: d_in/d_out are raw CUDA device addresses; deviceptr() tells OpenACC NOT to copy
! or map -- the data is already resident. The CUDA stream created by the C++ driver is bound to an
! OpenACC async queue via acc_set_cuda_stream, so the region runs ORDERED on the driver's stream.
module acc_interop
  use iso_c_binding
  implicit none
  interface
     ! C runtime entry: int acc_set_cuda_stream(int async, void* stream)
     function acc_set_cuda_stream(async, stream) result(res) bind(C, name="acc_set_cuda_stream")
       use iso_c_binding
       integer(c_int), value, intent(in) :: async
       type(c_ptr), value, intent(in) :: stream
       integer(c_int) :: res
     end function acc_set_cuda_stream
  end interface
end module acc_interop

subroutine acc_add1(d_in, d_out, n, stream) bind(C, name="acc_add1")
  use iso_c_binding
  use acc_interop
  implicit none
  type(c_ptr), value, intent(in) :: d_in, d_out
  integer(c_long), value, intent(in) :: n
  type(c_ptr), value, intent(in) :: stream
  real(c_double), pointer :: a(:), b(:)
  integer(c_int) :: qid, ierr
  integer(c_long) :: i

  call c_f_pointer(d_in, a, [n])
  call c_f_pointer(d_out, b, [n])

  qid = 1
  ierr = acc_set_cuda_stream(qid, stream)   ! bind async queue #1 to the driver's CUDA stream

  !$acc parallel loop deviceptr(a, b) async(qid)
  do i = 1, n
     b(i) = a(i) + 1.0d0
  end do
  ! no wait here: rely purely on stream ordering (the driver syncs the stream at the end)
end subroutine acc_add1
