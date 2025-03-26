!------------------------------------------------------
! Declaration of Precision of variables
MODULE precisions
  IMPLICIT NONE
  SAVE

  INTEGER, PARAMETER :: int_p = SELECTED_INT_KIND(8)
  INTEGER, PARAMETER :: real_p = SELECTED_REAL_KIND(8)

END MODULE precisions
!------------------------------------------------------

!------------------------------------------------------
! Definitions of parameters used in the geometry
MODULE geometry
  USE precisions
  IMPLICIT NONE
  SAVE
  
  INTEGER(int_p) :: n_c, n_v, n_f, n_f_b
  INTEGER(int_p), PARAMETER :: n = 3 ! number of cells along each dir.
  REAL(real_p), PARAMETER :: len = 0.9d0

  INTEGER(int_p), DIMENSION(:, :), POINTER :: lcf, lcv, lfv, lfc, snsign
  INTEGER(int_p), DIMENSION(:), POINTER :: bf_to_f, bface, bnode, f_to_bf
  REAL(real_p), DIMENSION(:), POINTER :: xc, yc, xv, yv, xf, yf, area, wfun, delta, vol, sumwt
  REAL(real_p), DIMENSION(:, :), POINTER :: sn, st, cwfun

! lcf   = cell to face connectivity
! lcv   = cell to vertex connectivity
! lfv   = face to vertex connectivity
! lfc   = face to cell connectivity
! area  = area of face
! vol   = volume of cell
! wfun  = cell-to-face interpolation function
! cwfun = cell-to-vertex interpolation function
! xc    = x coordinate of cell centroid
! yc    = y coordinate of cell centroid
! xv    = x coordinate of vertex
! yv    = y coordinate of vertex
! xf    = x coordinate of face
! yf    = y coordinate of face
! sn    = surface normal of face, has 3 components
! st    = surface tangent of face, has 3 components
! bf_to_f = boundary face number to global face number
! f_to_bf = global face number to boundary face number
! bface  = flag stating if boundary face (0 for interior faces)
! bnode  = flag stating if boundary node (0 for interior nodes)
! sn_sign = flag to indicate if surface normal is pointing in or out of cell

END MODULE geometry
!------------------------------------------------------
       program main
       
       IMPLICIT NONE
       
       CALL grid
       CALL solve

       END program main
!------------------------------------------------------
! Calculation of grid and all geometry related data
       
       SUBROUTINE grid
       
       USE precisions
       USE geometry
       
       IMPLICIT NONE
       INTEGER(int_p), PARAMETER :: info = 1
       INTEGER(int_p) :: i,j,ic,iv,ifc,ifc1,ifc2,ifc3,ifc4,ifc_b,jc,k, &
                        v1,v2,v3,v4,c1,c2
       REAL(real_p), PARAMETER :: one=1.0d0,two=2.0d0,four=4.0d0, &
                                threes=3.0d0,half=0.5d0,zero=0.0d0, nine=270.0d0
       REAL(real_p) :: d,pi,theta,dx1,dy1,dx2,dy2,dlen1,dlen2, &
                       d1,ndotl
                       
       pi = two*atan(one)
       theta = pi/threes    ! Angle of Rhombus
       d = len/n           ! size of each cell (assumed equal in x and y)

       ! Tally total number of cells
       ic = 0
       DO j = 1,n
          DO i = 1,n
             ic = ic + 1
          END DO
       END DO

       ! Tally total number of vertices (or nodes)
       iv = 0
       DO j = 1,n+1
          DO i = 1,n+1
             iv = iv + 1
          END DO
       END DO
       
       n_c = ic
       n_v = iv

       ALLOCATE(xc(n_c), yc(n_c))
       ALLOCATE(xv(n_v), yv(n_v))
       
! Cell center coordinates
       ic = 0
       DO j = 1, n
          DO i = 1, n
             ic = ic + 1
             xc(ic) = (i - 0.5d0)*d + (j - 0.5d0)*d*cos(theta)
             yc(ic) = (j - 0.5d0)*d*sin(theta)
          END DO
       END DO
       
       ! Vertex coordinates
       iv = 0
       DO j = 1, n+1
          DO i = 1, n+1
             iv = iv + 1
             xv(iv) = (i - 1)*d + (j - 1)*d*cos(theta)
             yv(iv) = (j - 1)*d*sin(theta)
          END DO
       END DO
       
       ! Grid Connectivity
       ! Cell to face connectivity
       ALLOCATE(lcf(n_c, 4))
       ic = 0
       DO j = 1, n
          DO i = 1, n
             ic = ic + 1
             lcf(ic, 1) = ic
             lcf(ic, 2) = n_c + n + ic + j
             lcf(ic, 3) = ic + n
             lcf(ic, 4) = n_c + n + ic + j - 1
          END DO
       END DO
       n_f = lcf(n_c, 2) ! Total number of faces

! Cell to vertex connectivity
       ALLOCATE(lcv(n_c, 4))
       ic = 0
       DO j = 1, n
          DO i = 1, n
             ic = ic + 1
             lcv(ic, 1) = ic + j - 1
             lcv(ic, 2) = ic + j
             lcv(ic, 3) = ic + n + j + 1
             lcv(ic, 4) = ic + n + j
          END DO
       END DO
       
       ! Face to Vertex Connectivity: Vertices are ordered counter-clockwise
       ALLOCATE(lfv(n_f, 2))
       DO ic = 1, n_c
          ! southern face vertices
          ifc1 = lcf(ic, 1)
          lfv(ifc1, 1) = lcv(ic, 1)
          lfv(ifc1, 2) = lcv(ic, 2)
       
          ! northern face vertices
          ifc2 = lcf(ic, 3)
          lfv(ifc2, 1) = lcv(ic, 3)
          lfv(ifc2, 2) = lcv(ic, 4)
       
          ! eastern face vertices
          ifc3 = lcf(ic, 2)
          lfv(ifc3, 1) = lcv(ic, 2)
          lfv(ifc3, 2) = lcv(ic, 3)
       
          ! western face vertices
          ifc4 = lcf(ic, 4)
          lfv(ifc4, 1) = lcv(ic, 4)
          lfv(ifc4, 2) = lcv(ic, 1)
       END DO
       
       ! Compute Face Center Coordinates
       ALLOCATE(xf(n_f), yf(n_f))
       DO ifc = 1, n_f
          v1 = lfv(ifc, 1)
          v2 = lfv(ifc, 2)
          xf(ifc) = half*(xv(v1) + xv(v2))
          yf(ifc) = half*(yv(v1) + yv(v2))
       END DO
       
! Boundary Face Indexing
       n_f_b = 2*(n+n)    ! Total number of boundary faces
       ALLOCATE(bf_to_f(n_f_b), f_to_bf(n_f))  
       ALLOCATE(bface(n_f))   ! Binary indicating if boundary face
       bface(:) = 0           ! Interior face by default
       f_to_bf(:) = 0
       ifc_b = 0
       
       ! Southern boundary faces
       DO i = 1, n
           ic = i
           ifc_b = ifc_b + 1
           bf_to_f(ifc_b) = lcf(ic, 1)
           bface(lcf(ic, 1)) = 1   ! Boundary face
           f_to_bf(lcf(ic, 1)) = ifc_b
       END DO
       
       ! Northern boundary faces
       DO i = 1, n
           ic = i + (n-1)*n
           ifc_b = ifc_b + 1
           bf_to_f(ifc_b) = lcf(ic, 3)
           bface(lcf(ic, 3)) = 1   ! Boundary face
           f_to_bf(lcf(ic, 3)) = ifc_b
       END DO
       
       ! Western boundary faces
       DO j = 1, n
           ic = 1 + (j-1)*n
           ifc_b = ifc_b + 1
           bf_to_f(ifc_b) = lcf(ic, 4)
           bface(lcf(ic, 4)) = 1   ! Boundary face
           f_to_bf(lcf(ic, 4)) = ifc_b
       END DO
       
       ! Eastern boundary faces
       DO j = 1, n
           ic = n + (j-1)*n
           ifc_b = ifc_b + 1
           bf_to_f(ifc_b) = lcf(ic, 2)
           bface(lcf(ic, 2)) = 1   ! Boundary face
           f_to_bf(lcf(ic, 2)) = ifc_b
       END DO

       !   19:24
! Face to Cell Connectivity
       ALLOCATE(lfc(n_f, 2))
       lfc(:, :) = -999
       
       DO ic = 1, n_c
           DO k = 1, 4
               ifc1 = lcf(ic, k)
               IF (lfc(ifc1, 1) /= -999) CYCLE
               lfc(ifc1, 1) = ic
               IF (bface(ifc1) == 1) THEN
                   lfc(ifc1, 2) = ic
               ELSE
                   IF (k == 1) THEN
                       jc = ic - n
                   ELSEIF (k == 3) THEN
                       jc = ic + n
                   ELSEIF (k == 2) THEN
                       jc = ic + 1
                   ELSEIF (k == 4) THEN
                       jc = ic - 1
                   END IF
                   lfc(ifc1, 2) = jc
               END IF
           END DO
       END DO

! Area, unit surface normal, and unit surface tangent
! Surface normal always pointed from cell1 to cell2
! At boundary, surface normal is pointed outward
! Surface tangent is pointed from vertex1 to vertex2

       ALLOCATE(area(n_f))
       ALLOCATE(sn(n_f, 2))
       ALLOCATE(st(n_f, 2))
       
       ! Compute sn and st first. The cross product is k.
       DO ifc = 1, n_f
           v1 = lfv(ifc, 1)
           v2 = lfv(ifc, 2)
       
           dx1 = xv(v2) - xv(v1)
           dy1 = yv(v2) - yv(v1)
       
           area(ifc) = SQRT(dx1*dx1 + dy1*dy1)
       
           sn(ifc, 1) =  dy1 / area(ifc)
           sn(ifc, 2) = -dx1 / area(ifc)
       
           st(ifc, 1) =  dx1 / area(ifc)
           st(ifc, 2) =  dy1 / area(ifc)
       END DO
       
       ! Make sure surface normal points from cell 1 to cell 2
       DO ifc = 1, n_f
           c1 = lfc(ifc, 1)
           c2 = lfc(ifc, 2)
       
           IF (c1 == c2) THEN ! Make external normals point out
               dx1 = xf(ifc) - xc(c1)
               dy1 = xf(ifc) - xc(c1)
               ndotl = dx1*sn(ifc, 1) + dy1*sn(ifc, 2)
               
               IF (ndotl < 0.0D0) THEN
                   sn(ifc, :) = -sn(ifc, :)
                   st(ifc, :) = -st(ifc, :)
               END IF
           ELSE ! Point from cell 1 to cell 2
               dx1 = xc(c2) - xc(c1)
               dy1 = xc(c2) - xc(c1)
               ndotl = dx1*sn(ifc, 1) + dy1*sn(ifc, 2)
               
               IF (ndotl < 0.0D0) THEN
                   sn(ifc, :) = -sn(ifc, :)
                   st(ifc, :) = -st(ifc, :)
               END IF
           END IF
       END DO

! Calculate delta = n . l
       ALLOCATE(delta(n_f))
       DO ifc = 1, n_f
           c1 = lfc(ifc, 1)
           c2 = lfc(ifc, 2)
           
           IF (c1 == c2) THEN ! Boundary faces
               dx1 = xf(ifc) - xc(c1)
               dy1 = yf(ifc) - yc(c1)
               delta(ifc) = ABS(dx1*sn(ifc, 1) + dy1*sn(ifc, 2))
           ELSE ! Interior faces
               dx1 = xc(c2) - xc(c1)
               dy1 = yc(c2) - yc(c1)
               delta(ifc) = ABS(dx1*sn(ifc, 1) + dy1*sn(ifc, 2))
           END IF
       END DO
       
       ! Make surface normal point outward from cell's perspective
       ALLOCATE(snsign(n_c, 4))
       DO ic = 1, n_c
           DO j = 1, 4  ! Loop over faces of cell (assuming quads)
               ifc = lcf(ic, j)
               c1 = lfc(ifc, 1)
               IF (ic == c1) THEN  ! Surface normal already out
                   snsign(ic, j) = 1
               ELSE
                   snsign(ic, j) = -1  ! Flip direction
               END IF
           END DO
       END DO
       
       ! 30:37
! Calculate cell-to-face interpolation weights
! Note that the interpolation function has been computed such that
! face_value = wfun*cell_value(1) + (1-wfun)*cell_value(2)
       ALLOCATE(wfun(n_f))
       wfun(:) = zero
       DO ifc = 1, n_f
           c1 = lfc(ifc, 1)
           c2 = lfc(ifc, 2)
           dx1 = xf(ifc) - xc(c1)
           dy1 = yf(ifc) - yc(c1)
           dlen1 = SQRT(dx1**2 + dy1**2)
           
           dx2 = xf(ifc) - xc(c2)
           dy2 = yf(ifc) - yc(c2)
           dlen2 = SQRT(dx2**2 + dy2**2)
           
           wfun(ifc) = dlen2 / (dlen1 + dlen2)
       END DO

       ! Calculate cell-to-vertex interpolation weights
       ! The interpolation weight is 1/distance
       ! Vertex value = SUM(cwfun*cell_value)
       ALLOCATE(cwfun(n_c, 4), sumwt(n_v))
       cwfun(:,:) = zero
       sumwt(:) = zero
       DO ic = 1, n_c
           DO iv = 1, 4  ! assuming 4 vertices
               v1 = lcv(ic, iv)
               d1 = SQRT((xv(v1)-xc(ic))**2 + (yv(v1)-yc(ic))**2)
               cwfun(ic, iv) = one/d1
               sumwt(v1) = sumwt(v1) + one/d1
           END DO
       END DO
       
       DO ic = 1, n_c
           DO iv = 1, 4
               v1 = lcv(ic, iv)
               cwfun(ic, iv) = cwfun(ic, iv)/sumwt(v1)
           END DO
       END DO

! Calculation of Cell Volume
       ALLOCATE(vol(n_c))
       vol(:) = zero
       DO ic = 1, n_c
           v1 = lcv(ic, 1)
           v2 = lcv(ic, 2)
           v3 = lcv(ic, 3)
           v4 = lcv(ic, 4)
           
           dx1 = xv(v1) - xv(v3)
           dy1 = yv(v1) - yv(v3)
           dx2 = xv(v2) - xv(v3)
           dy2 = yv(v2) - yv(v3)
           
           vol(ic) = half*(dx1*dy2 - dy1*dx2)
           
           dx1 = xv(v1) - xv(v4)
           dy1 = yv(v1) - yv(v4)
           dx2 = xv(v3) - xv(v4)
           dy2 = yv(v3) - yv(v4)
           
           vol(ic) = vol(ic) + half*(dx1*dy2 - dy1*dx2)
       END DO
       
       ! Tag Boundary Nodes
       ALLOCATE(bnode(n_v))
       bnode(:) = 0
       DO ifc = 1, n_f
           IF (bface(ifc) == 1) THEN
               DO j = 1, 2
                   v1 = lfv(ifc, j)
                   bnode(v1) = 1
               END DO
           END IF
       END DO
       

! Write out Grid Data for Cross-Check/Debugging
       IF(info > 0) THEN
        OPEN(unit = 10, file = "grid.dat", status = "unknown")
        
        WRITE(10,*) "Number of Cells =", n_c
        WRITE(10,*) "Number of Faces =", n_f
        WRITE(10,*) "Number of Vertices =", n_v
        WRITE(10,*) "Number of Boundary Faces =", n_f_b
    
        WRITE(10,*) "Cell Center Data"
        DO ic = 1, n_c
            WRITE(10,11) ic, xc(ic), yc(ic), vol(ic)
        END DO
    
        WRITE(10,*) "Face Center Data"
        DO ifc = 1, n_f
            WRITE(10,11) ifc, xf(ifc), yf(ifc), area(ifc)
        END DO
    
        WRITE(10,*) "Surface Normal Data"
        DO ifc = 1, n_f
            WRITE(10,11) ifc, xf(ifc), yf(ifc), sn(ifc,1), sn(ifc,2)
        END DO
    
        WRITE(10,*) "Delta"
        DO ifc = 1, n_f
            WRITE(10,11) ifc, xf(ifc), yf(ifc), delta(ifc)
        END DO
    
        WRITE(10,*) "Vertex Data"
        DO iv = 1, n_v
            WRITE(10,12) iv, xv(iv), yv(iv)
        END DO
    
        WRITE(10,*) "Cell to Face Connectivity"
        DO ic = 1, n_c
            WRITE(10,*) ic, (lcf(ic, ifc), ifc = 1, 4)
        END DO
    
        WRITE(10,*) "Face to Cell Connectivity"
        DO ifc = 1, n_f
            WRITE(10,*) ifc, (lfc(ifc, ic), ic = 1, 2)
        END DO
        
        CLOSE(unit = 10)
    END IF
       
11 format(i5,3(f13.6))
12 format(i5,2(f13.6))
13 format(i5,4(f13.6))

    END SUBROUTINE grid

    ! 35:27
!------------------------------------------------------
    SUBROUTINE solve
        USE precisions
        USE geometry
      
        IMPLICIT NONE
        INTEGER(int_p) :: ic, ifc, iv, ifb, j, icn, iter, c1, c2, v1, v2
        INTEGER(int_p), PARAMETER :: max_iter = 100000
        REAL(real_p), DIMENSION(:), POINTER :: ap, sc, phib, phi, phinode, weight, scskew
        REAL(real_p), DIMENSION(:,:), POINTER :: anb
        REAL(real_p) :: sumf, sumr, res, dx1, dy1, tdotl
        REAL(real_p), PARAMETER :: tol = 1.0e-8, phi_bot = 1.0d0

        ALLOCATE(ap(n_c), sc(n_c), anb(n_c, 4), phib(n_f_b), phi(n_c))
        ALLOCATE(phinode(n_v), weight(n_v))
        ALLOCATE(scskew(n_c))
        phi(:) = 0.0d0
      
        ! Open Residual File
        OPEN(unit=30, file="Ex7.1.rs", status="unknown")
      
        ! Hardwire BC at bottom wall
        phib(:) = 0.0d0
        DO ifb = 1, n_f_b
          ifc = bf_to_f(ifb)
          IF(yf(ifc) < 1.00-3) THEN  ! Bottom wall
            phib(ifb) = phi_bot
          ENDIF
        ENDDO

        PRINT *, "Array phib:"
        DO ic = 1, ifb
          PRINT "(F12.6)", phib(ic)
        ENDDO
      
        ! Calculate Link Coefficients
        DO ic = 1, n_c
          ap(ic) = 0.0d0
          sc(ic) = 0.0d0
          DO j = 1, 4
            ifc = lcf(ic,j)
            IF(bface(ifc) == 0) THEN  ! Interior faces
              ap(ic) = ap(ic) + area(ifc) / delta(ifc)
              anb(ic,j) = -area(ifc) / delta(ifc)
            ELSE  ! Boundary faces
              ifb = f_to_bf(ifc)
              ap(ic) = ap(ic) + area(ifc) / delta(ifc)
              anb(ic,j) = 0.0d0
              sc(ic) = sc(ic) + phib(ifb) * area(ifc) / delta(ifc)
            ENDIF
          ENDDO
        ENDDO
      
        PRINT *, "Array anb:"
        DO ic = 1, n_c
          PRINT "(4F12.6)", (anb(ic,j), j=1,4)
        ENDDO

        PRINT *, "Array sc:"
        DO ic = 1, n_c
          PRINT "(F12.6)", sc(ic)
        ENDDO
      
! Start of Gauss-Seidel Iteration
        DO iter = 1, max_iter
            ! Compute vertex values
            phinode(:) = 0.0d0; weight(:) = 0.0d0
            DO ic = 1, n_c
              DO j = 1, 4
                iv = lcv(ic, j)
                IF(bnode(iv) == 1) CYCLE
                weight(iv) = weight(iv) + cwfun(ic, j)
                phinode(iv) = phinode(iv) + phi(ic) * cwfun(ic, j)
              ENDDO
            ENDDO
          
            ! Hardwire lower boundary nodes
            DO iv = 1, n_v
              IF(bnode(iv) == 1) THEN
                IF(yv(iv) < 1.0d-3) phinode(iv) = phi_bot
              ENDIF
            ENDDO
          
! Compute tangential flux (skew) source
          DO ic = 1, n_c
            scskew(ic) = 0.0d0
            sumf = 0.0d0
            DO j = 1, 4
              ifc = lcf(ic, j)
              IF(bface(ifc) == 1) CYCLE
              c1 = lfc(ifc, 1)
              c2 = lfc(ifc, 2)
              v1 = lfv(ifc, 1)
              v2 = lfv(ifc, 2)
              dx1 = xc(c2) - xc(c1)
              dy1 = yc(c2) - yc(c1)
              tdotl = st(ifc, 1) * dx1 + st(ifc, 2) * dy1
              sumf = sumf + tdotl * (phinode(v2) - phinode(v1)) * snsign(ic, j) / delta(ifc)
            ENDDO
            scskew(ic) = sumf
          ENDDO
          
          ! Update solution using GS formula
          DO ic = 1, n_c
            sumf = 0.0d0
            DO j = 1, 4  ! Summing over all neighbors
              ifc = lcf(ic, j)
              IF(snsign(ic, j) == 1) THEN  ! Figuring out neighboring cell index
                icn = lfc(ifc, 2)
              ELSE
                icn = lfc(ifc, 1)
              ENDIF
              sumf = sumf + anb(ic, j) * phi(icn)
            ENDDO
            phi(ic) = (sc(ic) + scskew(ic) - sumf) / ap(ic)
          ENDDO

        
! Compute Residual
          sumr = 0.0d0
          DO ic = 1, n_c
            sumf = 0.0d0
            DO j = 1, 4  ! Summing over all neighbors
              ifc = lcf(ic, j)
              IF(snsign(ic, j) == 1) THEN  ! Figuring out neighboring cell index
                icn = lfc(ifc, 2)
              ELSE
                icn = lfc(ifc, 1)
              ENDIF
              sumf = sumf + anb(ic, j) * phi(icn)
            ENDDO
            sumr = sumr + (sc(ic) + scskew(ic) - ap(ic) * phi(ic) - sumf)**2
          ENDDO
          res = SQRT(MAX(0.0d0, sumr))  ! L2 norm
          
          ! PRINT *, iter, res
          WRITE(30,*) iter, res
          
          IF(res < tol) EXIT
          
        ENDDO  ! Iteration loop
          
          ! Compute Vertex Values
          phinode(:) = 0.0d0; weight(:) = 0.0d0
          DO ic = 1, n_c
            DO j = 1, 4
              iv = lcv(ic, j)
              IF(bnode(iv) == 1) CYCLE
              weight(iv) = weight(iv) + cwfun(ic, j)
              phinode(iv) = phinode(iv) + phi(ic) * cwfun(ic, j)
            ENDDO
          ENDDO
          
          ! Hardwire lower boundary nodes
          DO iv = 1, n_v
            IF(bnode(iv) == 1) THEN
              IF(yv(iv) <= 1.0d-3) phinode(iv) = phi_bot
            ENDIF
          ENDDO

          
! Write solution for TECPLOT in FE format
          OPEN(unit=20, file="contour.dat", status="unknown")
          WRITE(20,*) 'VARIABLES = "X", "Y", "PHI"'
          WRITE(20,*) 'ZONE N=',n_v,', E=',n_c,', DATAPACKING=POINT, ZONETYPE=FEQUADRILATERAL'
          
          DO iv = 1, n_v
            WRITE(20,*) xv(iv), yv(iv), phinode(iv)
          ENDDO
          
          DO ic = 1, n_c
            WRITE(20,*) (lcv(ic,j), j=1,4)
          ENDDO
          
11          FORMAT(i5,3(F13.6))
          
        END SUBROUTINE solve
          