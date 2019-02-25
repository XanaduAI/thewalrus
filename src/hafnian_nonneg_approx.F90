! Copyright 2019 Xanadu Quantum Technologies Inc.
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.



module hafnian_approx
  use kinds
  use vars
  use omp_lib
  use ISO_FORTRAN_ENV
  use linpack_q_complex

  implicit none

  contains

  subroutine hafnian_nonneg(matin, nsample, haf)
    use kinds
    use vars
    use omp_lib
    use linpack_q_complex

    implicit none

    real(dp), intent(in) :: matin(:, :)
    integer(ip), intent(in) :: nsample
    real(wp), allocatable :: mat(:, :), matrand(:, :), matrand1(:, :), matrand2(:, :), g(:,:), gt(:,:), &
                             determinants(:), W(:,:)
    real(dp), intent(out) :: haf

    integer(ip) :: n, i, j, k
    real(wp) :: det, rand1, rand2, std, mean

    !f2py intent(in) :: matin, nsample
    !f2py intent(out) :: haf

    std = 1.0_wp
    mean = 0.0_wp
    n = size(matin(1,:))

    allocate(mat(1:n, 1:n), matrand(1:n, 1:n), matrand1(1:n, 1:n), matrand2(1:n, 1:n), &
             determinants(1:nsample), g(1:n, 1:n), gt(1:n, 1:n), W(1:n, 1:n))

    forall (i=1:n, j=1:n) mat(i,j) = real(sqrt(real(matin(i,j), wp)), wp)
    determinants(:) = 0.0_wp

    !$OMP PARALLEL DO private(i,j,k,matrand1,matrand2,g,gt,matrand,W,det) shared(mat,determinants,std,mean)
    do k=1, nsample
      matrand(:,:) = 0.0_wp
      g(:,:) = 0.0_wp
      gt(:,:) = 0.0_wp

      call random_number(matrand1)
      call random_number(matrand2)

      forall (i=1:n, j=1:n) matrand(i,j) = std * sqrt(-2.0_wp*log(matrand1(i,j))) * &
                                          sin(2.0_wp*pi*matrand2(i,j)) + mean


      forall (i=1:n) g(i,i:n) = matrand(i,i:n)
      gt = transpose(g)
      forall (i=1:n, j=1:n) W(i,j) = (g(i,j)-gt(i,j))*mat(i,j)

      call det_real(W, det)
  
      determinants(k) = det
    end do
    !$OMP END PARALLEL DO

    haf = real(sum(determinants(:)), dp)/nsample


    deallocate(mat, matrand, g, gt, determinants, W)

  end subroutine hafnian_nonneg

  subroutine det_real(matin, det_out)
    use kinds
    use vars
    use omp_lib
    use ISO_FORTRAN_ENV

    implicit none

    ! determinant work variables
    real(wp)                  :: det(2), tmpdet
    integer(kind=4)              :: info, lda, nn, job, i, j
    integer(kind=4), allocatable :: ipvt(:)

    real(wp), allocatable    :: work(:)
    integer(ip)  :: n

    real(wp), intent(in) :: matin(:, :)
    real(wp), allocatable :: mat(:, :)
    real(wp), intent(out) :: det_out

    nn = size(matin(1,:))
    lda = nn

    allocate(ipvt(1:nn), work(1:nn), mat(1:nn,1:nn))

    forall (i=1:nn,j=1:nn) mat(i,j) = real(matin(i,j), wp)


    call qgefa(mat, lda, nn, ipvt, info)
    job = 10
    call qgedi(mat, lda, nn, ipvt, det, work, job )


    tmpdet = det(1) * 10.0_wp**(real(det(2), wp))

    det_out = real(tmpdet, wp)

    deallocate(ipvt, work, mat)

  end subroutine det_real

end module hafnian_approx
