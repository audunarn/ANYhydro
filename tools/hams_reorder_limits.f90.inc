! SPDX-License-Identifier: MPL-2.0
! ANYhydro opt-in GNU producer I/O replacement; upstream source notices retained.
      SUBROUTINE ReorderAmssDamp(filename, NFREQ, NDOF)
          USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
          USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: IOSTAT_END, IOSTAT_EOR
          IMPLICIT NONE
          CHARACTER(LEN=*), INTENT(IN) :: filename
          INTEGER, INTENT(IN) :: NFREQ, NDOF
          INTEGER :: IOS, I, J, K, IFREQ, ROW_VAL, COL_VAL, WIDTH, ORIGINAL_SIZE
          REAL*8 :: FREQ_VAL, AMAS_VAL, BDMP_VAL, LAST_FREQ
          REAL*8, ALLOCATABLE :: AMAS(:,:,:), BDMP(:,:,:), FREQ_LIST(:)
          LOGICAL, ALLOCATABLE :: SEEN(:,:,:), LIMIT_ROW(:)
          LOGICAL :: EXISTS, FILE_OPEN
          CHARACTER(LEN=55) :: LINE
          CHARACTER(LEN=:), ALLOCATABLE :: STAGED, BACKUP, ORIGINAL_BYTES

          IF (NFREQ < 1 .OR. NDOF < 1) ERROR STOP 'Invalid reorder inventory'
          INQUIRE(UNIT=61, OPENED=FILE_OPEN)
          IF (FILE_OPEN) THEN
              CLOSE(61, IOSTAT=IOS)
              IF (IOS /= 0) ERROR STOP 'Cannot close native radiation writer'
          ENDIF
          ALLOCATE(AMAS(NFREQ,NDOF,NDOF), BDMP(NFREQ,NDOF,NDOF), &
              SEEN(NFREQ,NDOF,NDOF), FREQ_LIST(NFREQ), LIMIT_ROW(NFREQ), STAT=IOS)
          IF (IOS /= 0) ERROR STOP 'Cannot allocate reorder inventory'
          SEEN=.FALSE.
          IFREQ=0
          LAST_FREQ=-999.D0
          OPEN(UNIT=901, FILE=TRIM(filename), STATUS='OLD', ACTION='READ', IOSTAT=IOS)
          IF (IOS /= 0) ERROR STOP 'Cannot read native radiation table'
          DO
              ! Width55 detects an overlong record instead of truncating it.
              ! A native record has exactly40(limit) or54(finite) columns.
              READ(901,'(A)',ADVANCE='NO',IOSTAT=IOS,SIZE=WIDTH) LINE
              IF (IOS == IOSTAT_END) EXIT
              IF (IOS /= IOSTAT_EOR) ERROR STOP 'Invalid native radiation record length'
              IF (LEN_TRIM(LINE(1:14)) == 0 .OR. LEN_TRIM(LINE(15:20)) == 0 &
                  .OR. LEN_TRIM(LINE(21:26)) == 0 .OR. LEN_TRIM(LINE(27:40)) == 0) &
                  ERROR STOP 'Missing native radiation numeric field'
              IF (.NOT.REAL_FIELD_OK(LINE(1:14)) .OR. .NOT.INTEGER_FIELD_OK(LINE(15:20)) &
                  .OR. .NOT.INTEGER_FIELD_OK(LINE(21:26)) .OR. .NOT.REAL_FIELD_OK(LINE(27:40))) &
                  ERROR STOP 'Invalid native radiation literal syntax'
              IF (WIDTH == 40) THEN
                  READ(LINE(1:40),'(ES14.6,2I6,ES14.6)',IOSTAT=IOS) &
                      FREQ_VAL, ROW_VAL, COL_VAL, AMAS_VAL
                  BDMP_VAL=0.D0 ! Scratch only; absence is preserved on output.
              ELSEIF (WIDTH == 54) THEN
                  IF (LEN_TRIM(LINE(41:54)) == 0) ERROR STOP 'Missing native radiation damping field'
                  IF (.NOT.REAL_FIELD_OK(LINE(41:54))) ERROR STOP 'Invalid native damping literal syntax'
                  READ(LINE(1:54),'(ES14.6,2I6,2ES14.6)',IOSTAT=IOS) &
                      FREQ_VAL, ROW_VAL, COL_VAL, AMAS_VAL, BDMP_VAL
              ELSE
                  ERROR STOP 'Invalid native radiation field inventory'
              ENDIF
              IF (IOS /= 0) ERROR STOP 'Invalid native radiation numeric fields'
              IF (.NOT.IEEE_IS_FINITE(FREQ_VAL) .OR. .NOT.IEEE_IS_FINITE(AMAS_VAL) &
                  .OR. .NOT.IEEE_IS_FINITE(BDMP_VAL)) ERROR STOP 'Nonfinite native radiation value'
              IF (WIDTH == 40) THEN
                  IF (FREQ_VAL /= 0.D0 .AND. FREQ_VAL /= -1.D0) &
                      ERROR STOP 'Missing finite-frequency damping'
              ELSE
                  IF (FREQ_VAL <= 0.D0) ERROR STOP 'Limit damping is not a computed coefficient'
              ENDIF
              IF (ROW_VAL < 1 .OR. ROW_VAL > NDOF .OR. COL_VAL < 1 .OR. COL_VAL > NDOF) &
                  ERROR STOP 'Radiation DOF outside declared inventory'
              ! Printed identities are exact. A collision produces a duplicate,
              ! rather than being merged using an arbitrary frequency tolerance.
              IF (IFREQ == 0 .OR. FREQ_VAL /= LAST_FREQ) THEN
                  IF (IFREQ > 0) THEN
                      IF (.NOT.ALL(SEEN(IFREQ,:,:))) ERROR STOP 'Incomplete native frequency block'
                      IF (ANY(FREQ_LIST(1:IFREQ) == FREQ_VAL)) &
                          ERROR STOP 'Noncontiguous repeated native frequency'
                  ENDIF
                  IFREQ=IFREQ+1
                  IF (IFREQ > NFREQ) ERROR STOP 'Extra native frequency block'
                  FREQ_LIST(IFREQ)=FREQ_VAL
                  LIMIT_ROW(IFREQ)=(WIDTH == 40)
                  LAST_FREQ=FREQ_VAL
              ENDIF
              IF (SEEN(IFREQ,ROW_VAL,COL_VAL)) ERROR STOP 'Duplicate native radiation coefficient'
              SEEN(IFREQ,ROW_VAL,COL_VAL)=.TRUE.
              AMAS(IFREQ,ROW_VAL,COL_VAL)=AMAS_VAL
              BDMP(IFREQ,ROW_VAL,COL_VAL)=BDMP_VAL
          ENDDO
          CLOSE(901,IOSTAT=IOS)
          IF (IOS /= 0) ERROR STOP 'Cannot close native radiation input'
          IF (IFREQ /= NFREQ .OR. .NOT.ALL(SEEN)) ERROR STOP 'Incomplete native radiation inventory'

          STAGED=TRIM(filename)//'.reordered'
          BACKUP=TRIM(filename)//'.before-reorder'
          INQUIRE(FILE=STAGED,EXIST=EXISTS)
          IF (EXISTS) ERROR STOP 'Reorder staging path already exists'
          INQUIRE(FILE=BACKUP,EXIST=EXISTS)
          IF (EXISTS) ERROR STOP 'Reorder original backup already exists'
          OPEN(UNIT=902,FILE=STAGED,STATUS='NEW',ACTION='WRITE',IOSTAT=IOS)
          IF (IOS /= 0) ERROR STOP 'Cannot stage reordered radiation table'
          DO I=1,NFREQ
              DO J=1,NDOF
                  DO K=1,NDOF
                      IF (LIMIT_ROW(I)) THEN
                          WRITE(902,'(ES14.6,2I6,ES14.6)',IOSTAT=IOS) &
                              FREQ_LIST(I),J,K,AMAS(I,J,K)
                      ELSE
                          WRITE(902,'(ES14.6,2I6,2ES14.6)',IOSTAT=IOS) &
                              FREQ_LIST(I),J,K,AMAS(I,J,K),BDMP(I,J,K)
                      ENDIF
                      IF (IOS /= 0) ERROR STOP 'Cannot write staged radiation coefficient'
                  ENDDO
              ENDDO
          ENDDO
          CLOSE(902,IOSTAT=IOS)
          IF (IOS /= 0) ERROR STOP 'Cannot close staged radiation table'

          ! Preserve original bytes through write/close/rename failures. These
          ! extra files are incomplete evidence until the consumer seals a run.
          INQUIRE(FILE=TRIM(filename),SIZE=ORIGINAL_SIZE,IOSTAT=IOS)
          IF (IOS /= 0 .OR. ORIGINAL_SIZE < 1) ERROR STOP 'Cannot size original radiation table'
          ALLOCATE(CHARACTER(LEN=ORIGINAL_SIZE)::ORIGINAL_BYTES,STAT=IOS)
          IF (IOS /= 0) ERROR STOP 'Cannot allocate original radiation backup'
          OPEN(UNIT=903,FILE=TRIM(filename),STATUS='OLD',ACCESS='STREAM',FORM='UNFORMATTED',ACTION='READ',IOSTAT=IOS)
          IF (IOS /= 0) ERROR STOP 'Cannot open original radiation bytes'
          READ(903,IOSTAT=IOS) ORIGINAL_BYTES
          IF (IOS /= 0) ERROR STOP 'Cannot read original radiation bytes'
          CLOSE(903,IOSTAT=IOS)
          IF (IOS /= 0) ERROR STOP 'Cannot close original radiation bytes'
          OPEN(UNIT=904,FILE=BACKUP,STATUS='NEW',ACCESS='STREAM',FORM='UNFORMATTED',ACTION='WRITE',IOSTAT=IOS)
          IF (IOS /= 0) ERROR STOP 'Cannot create original radiation backup'
          WRITE(904,IOSTAT=IOS) ORIGINAL_BYTES
          IF (IOS /= 0) ERROR STOP 'Cannot write original radiation backup'
          CLOSE(904,IOSTAT=IOS)
          IF (IOS /= 0) ERROR STOP 'Cannot close original radiation backup'
          ! GNU intrinsic uses the native same-directory rename operation.
          CALL RENAME(STAGED,TRIM(filename),IOS)
          IF (IOS /= 0) ERROR STOP 'Cannot publish staged radiation table; original retained'
          WRITE(*,*) ' AmssDamp.1 has been rearranged with verified native inventory.'
      CONTAINS
          LOGICAL FUNCTION DIGITS(TEXT)
              CHARACTER(*), INTENT(IN) :: TEXT
              INTEGER :: POS
              DIGITS=.FALSE.
              IF (LEN(TEXT) < 1) RETURN
              DO POS=1,LEN(TEXT)
                  IF (TEXT(POS:POS) < '0' .OR. TEXT(POS:POS) > '9') RETURN
              ENDDO
              DIGITS=.TRUE.
          END FUNCTION DIGITS
          LOGICAL FUNCTION INTEGER_FIELD_OK(TEXT)
              CHARACTER(*), INTENT(IN) :: TEXT
              CHARACTER(:), ALLOCATABLE :: TOKEN
              INTEGER :: POS
              INTEGER_FIELD_OK=.FALSE.
              IF (TEXT(LEN(TEXT):LEN(TEXT)) == ' ') RETURN
              TOKEN=TRIM(ADJUSTL(TEXT));POS=1
              IF (LEN(TOKEN) < 1) RETURN
              IF (TOKEN(1:1) == '+' .OR. TOKEN(1:1) == '-') POS=2
              INTEGER_FIELD_OK=DIGITS(TOKEN(POS:))
          END FUNCTION INTEGER_FIELD_OK
          LOGICAL FUNCTION REAL_FIELD_OK(TEXT)
              CHARACTER(*), INTENT(IN) :: TEXT
              CHARACTER(:), ALLOCATABLE :: TOKEN
              INTEGER :: POS, EXP_START, EXP_DIGITS
              LOGICAL :: EXPLICIT_E
              REAL_FIELD_OK=.FALSE.
              IF (TEXT(LEN(TEXT):LEN(TEXT)) == ' ') RETURN
              TOKEN=TRIM(ADJUSTL(TEXT));POS=1
              IF (LEN(TOKEN) < 1) RETURN
              IF (TOKEN(1:1) == '+' .OR. TOKEN(1:1) == '-') POS=2
              IF (LEN(TOKEN) < POS+10) RETURN
              IF (.NOT.DIGITS(TOKEN(POS:POS)) .OR. TOKEN(POS+1:POS+1) /= '.') RETURN
              IF (.NOT.DIGITS(TOKEN(POS+2:POS+7))) RETURN
              EXP_START=POS+8;EXPLICIT_E=(TOKEN(EXP_START:EXP_START) == 'E')
              IF (EXPLICIT_E) EXP_START=EXP_START+1
              IF (TOKEN(EXP_START:EXP_START) /= '+' .AND. TOKEN(EXP_START:EXP_START) /= '-') RETURN
              EXP_START=EXP_START+1;EXP_DIGITS=LEN(TOKEN)-EXP_START+1
              ! GNU ES14.6 can omit E for a three-digit exponent.
              IF (EXPLICIT_E) THEN
                  IF (EXP_DIGITS /= 2 .AND. EXP_DIGITS /= 3) RETURN
              ELSE
                  IF (EXP_DIGITS /= 3) RETURN
              ENDIF
              REAL_FIELD_OK=DIGITS(TOKEN(EXP_START:))
          END FUNCTION REAL_FIELD_OK
      END SUBROUTINE ReorderAmssDamp
