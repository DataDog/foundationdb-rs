use super::*;
use memchr::memchr_iter;
use std::convert::TryFrom;
use std::io;
use std::mem;

trait VersionstampState {}

const PACK_ERR_MSG: &str = "pack io error on Vec, data size didn't fit in `u32`?";

trait WriteAll {
    fn write_all(&mut self, bytes: &[u8]) -> io::Result<()>;
}

#[derive(Debug, Clone, Copy)]
struct Incomplete {
    offset: u32,
}

#[derive(Debug, Clone, Copy)]
struct Complete {
    size: u32,
}

impl VersionstampState for Incomplete {}
impl VersionstampState for Complete {}

struct PackedTuple<W: io::Write, T: VersionstampState> {
    writer: W,
    versionstamp_state: T,
}

impl<W: io::Write> PackedTuple<W, Incomplete> {
    #[must_use]
    fn pack(mut self) -> io::Result<()> {
        let offset = self.versionstamp_state.offset.to_le_bytes();
        self.write_all(&offset)?;
        Ok(())
    }
}

impl<W: io::Write> WriteAll for PackedTuple<W, Incomplete> {
    fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.writer.write_all(bytes)?;
        Ok(())
    }
}

impl<W: io::Write> PackedTuple<W, Complete> {
    fn into_incomplete(self) -> PackedTuple<W, Incomplete> {
        let versionstamp_state = Incomplete {
            offset: self.versionstamp_state.size,
        };
        PackedTuple {
            writer: self.writer,
            versionstamp_state,
        }
    }

    fn into_incomplete_with_offset(self, offset: u32) -> PackedTuple<W, Incomplete> {
        let versionstamp_state = Incomplete { offset };
        PackedTuple {
            writer: self.writer,
            versionstamp_state,
        }
    }
}

impl<W: io::Write> WriteAll for PackedTuple<W, Complete> {
    fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.versionstamp_state.size += bytes.len() as u32;
        self.writer.write_all(bytes)?;
        Ok(())
    }
}

/// A type that can be packed
pub trait TuplePack<I: VersionstampState, O: VersionstampState> {
    fn pack<W: io::Write>(
        &self,
        w: PackedTuple<W, I>,
        tuple_depth: TupleDepth,
    ) -> io::Result<PackedTuple<W, O>>;

    fn pack_root<W: io::Write>(&self, w: PackedTuple<W, I>) -> io::Result<PackedTuple<W, O>> {
        self.pack(w, TupleDepth::new())
    }
}

/// Pack value and returns the packed buffer
///
/// # Panics
///
/// Panics if the encoded data size doesn't fit in `u32`.
fn pack_to_vec<T: TuplePack<Complete, O>, O: VersionstampState>(
    tuple: &T,
) -> io::Result<PackedTuple<Vec<u8>, O>> {
    pack_into_vec(tuple, Vec::new())
}

/// Pack value into the given buffer
///
/// # Panics
///
/// Panics if the encoded data size doesn't fit in `u32`.
fn pack_into_vec<T: TuplePack<Complete, O>, O: VersionstampState>(
    tuple: &T,
    output: Vec<u8>,
) -> io::Result<PackedTuple<Vec<u8>, O>> {
    tuple.pack_root(PackedTuple {
        writer: output,
        versionstamp_state: Complete { size: 0 },
    })
}

/// Incomplete to Incomplete transition is always available for Complete to Complete transitions.
impl<T> TuplePack<Incomplete, Incomplete> for T
where
    T: TuplePack<Complete, Complete>,
{
    fn pack<W: io::Write>(
        &self,
        mut w: PackedTuple<W, Incomplete>,
        tuple_depth: TupleDepth,
    ) -> io::Result<PackedTuple<W, Incomplete>> {
        let offset = w.versionstamp_state.offset;

        // Using a placeholder packer for the Complete -> Complete state.
        let packer = PackedTuple {
            writer: w.writer,
            versionstamp_state: Complete { size: 0 },
        };

        let packer = self.pack(packer, tuple_depth)?;
        Ok(packer.into_incomplete_with_offset(offset))
    }
}

struct IncompleteVersionstamp {
    user_version: u16,
}

/// Complete to incomplete transition is only available to IncompleteVersionstamps.
impl TuplePack<Complete, Incomplete> for IncompleteVersionstamp {
    fn pack<W: io::Write>(
        &self,
        mut w: PackedTuple<W, Complete>,
        tuple_depth: TupleDepth,
    ) -> io::Result<PackedTuple<W, Incomplete>> {
        let mut bytes = [0xff; 12];
        bytes[10..].copy_from_slice(&self.user_version.to_be_bytes());
        w.write_all(&[VERSIONSTAMP])?;
        w.write_all(&bytes)?;
        Ok(w.into_incomplete())
    }
}

/// A type that can be unpacked
pub trait TupleUnpack<'de>: Sized {
    fn unpack(input: &'de [u8], tuple_depth: TupleDepth) -> PackResult<(&'de [u8], Self)>;

    fn unpack_root(input: &'de [u8]) -> PackResult<Self> {
        let (input, this) = Self::unpack(input, TupleDepth::new())?;
        if !input.is_empty() {
            return Err(PackError::TrailingBytes);
        }
        Ok(this)
    }
}

#[inline]
fn parse_bytes(input: &[u8], num: usize) -> PackResult<(&[u8], &[u8])> {
    if input.len() < num {
        Err(PackError::MissingBytes)
    } else {
        Ok((&input[num..], &input[..num]))
    }
}

#[inline]
fn parse_byte(input: &[u8]) -> PackResult<(&[u8], u8)> {
    if input.is_empty() {
        Err(PackError::MissingBytes)
    } else {
        Ok((&input[1..], input[0]))
    }
}

fn parse_code(input: &[u8], expected: u8) -> PackResult<&[u8]> {
    let (input, found) = parse_byte(input)?;
    if found == expected {
        Ok(input)
    } else {
        Err(PackError::BadCode {
            found,
            expected: Some(expected),
        })
    }
}

fn parse_slice(input: &[u8]) -> PackResult<(&[u8], Cow<[u8]>)> {
    let mut bytes = Vec::new();
    let mut pos = 0;
    for idx in memchr_iter(NIL, input) {
        let next_idx = idx + 1;
        if input.get(next_idx) == Some(&ESCAPE) {
            bytes.extend_from_slice(&input[pos..next_idx]);
            pos = next_idx + 1;
        } else {
            let slice = &input[pos..idx];
            return Ok((
                &input[next_idx..],
                (if pos == 0 {
                    Cow::Borrowed(slice)
                } else {
                    bytes.extend_from_slice(slice);
                    Cow::Owned(bytes)
                }),
            ));
        }
    }
    Err(PackError::MissingBytes)
}

fn parse_string(input: &[u8]) -> PackResult<(&[u8], Cow<str>)> {
    let (input, slice) = parse_slice(input)?;
    Ok((
        input,
        match slice {
            Cow::Borrowed(slice) => {
                Cow::Borrowed(std::str::from_utf8(slice).map_err(|_| PackError::BadStringFormat)?)
            }
            Cow::Owned(vec) => {
                Cow::Owned(String::from_utf8(vec).map_err(|_| PackError::BadStringFormat)?)
            }
        },
    ))
}

fn write_bytes<'a, W: io::Write, T: VersionstampState>(
    w: &mut PackedTuple<W, T>,
    v: &'a [u8],
) -> io::Result<()>
where
    PackedTuple<W, T>: WriteAll,
{
    u32::try_from(v.len()).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    let mut pos = 0;
    for idx in memchr_iter(NIL, v) {
        let next_idx = idx + 1;
        w.write_all(&v[pos..next_idx])?;
        w.write_all(&[ESCAPE])?;
        pos = next_idx;
    }
    w.write_all(&v[pos..])?;
    w.write_all(&[NIL])?;
    Ok(())
}

impl TuplePack<Complete, Complete> for () {
    fn pack<W: io::Write>(
        &self,
        mut w: PackedTuple<W, Complete>,
        tuple_depth: TupleDepth,
    ) -> io::Result<PackedTuple<W, Complete>> {
        if tuple_depth.depth() > 0 {
            w.write_all(&[NESTED, NIL])?;
            Ok(w)
        } else {
            Ok(w)
        }
    }
}

impl<'de> TupleUnpack<'de> for () {
    fn unpack(mut input: &[u8], tuple_depth: TupleDepth) -> PackResult<(&[u8], Self)> {
        if tuple_depth.depth() > 0 {
            input = parse_code(input, NESTED)?;
            input = parse_code(input, NIL)?;
        }
        Ok((input, ()))
    }
}

macro_rules! tuple_impls {
    ($((($n0:tt $name0:ident $v0:ident $o0:ident) $(($n:tt $name:ident $v:ident $i:ident $o:ident))* $out:ident))+) => {
        $(
            impl<$name0, $($name, $i: VersionstampState),*> TuplePack<Complete, $out> for ($name0, $($name,)*)
            where
                $name0: TuplePack<Complete, $o0>,
                $($name: TuplePack<$i, $o>,)*
            {
                fn pack<W: io::Write>(&self, mut w: PackedTuple<W, Complete>, tuple_depth: TupleDepth) -> io::Result<PackedTuple<W, $out>> {
                    if tuple_depth.depth() > 0 {
                        w.write_all(&[NESTED])?
                    }

                    w = self.$n0.pack(w, tuple_depth.increment())?;
                    $(
                        w = self.$n.pack(w, tuple_depth.increment())?;
                    )*

                    if tuple_depth.depth() > 0 {
                        w.write_all(&[NIL])?;
                    }
                    Ok(w)
                }
            }

            impl<'de, $name0, $($name),*> TupleUnpack<'de> for ($name0, $($name,)*)
            where
                $name0: TupleUnpack<'de>,
                $($name: TupleUnpack<'de>,)*
            {
                fn unpack(input: &'de [u8], tuple_depth: TupleDepth) -> PackResult<(&'de [u8], Self)> {
                    let input = if tuple_depth.depth() > 0 { parse_code(input, NESTED)? } else { input };

                    let (input, $v0) = $name0::unpack(input, tuple_depth.increment())?;
                    $(
                        let (input, $v) = $name::unpack(input, tuple_depth.increment())?;
                    )*

                        let input = if tuple_depth.depth() > 0 { parse_code(input, NIL)? } else { input };

                    let tuple = ( $v0, $($v,)* );
                    Ok((input, tuple))
                }
            }
        )+
    }
}

tuple_impls! {
((0 T0 t0 Complete) Complete)
((0 T0 t0 V1) (1 T1 t1 V1 Complete) Complete)
((0 T0 t0 V1) (1 T1 t1 V1 V2) (2 T2 t2 V2 Complete) Complete)
((0 T0 t0 V1) (1 T1 t1 V1 V2) (2 T2 t2 V2 V3) (3 T3 t3 V3 Complete) Complete)
((0 T0 t0 V1) (1 T1 t1 V1 V2) (2 T2 t2 V2 V3) (3 T3 t3 V3 V4) (4 T4 t4 V4 Complete) Complete)
((0 T0 t0 V1) (1 T1 t1 V1 V2) (2 T2 t2 V2 V3) (3 T3 t3 V3 V4) (4 T4 t4 V4 V5) (5 T5 t5 V5 Complete) Complete)
((0 T0 t0 V1) (1 T1 t1 V1 V2) (2 T2 t2 V2 V3) (3 T3 t3 V3 V4) (4 T4 t4 V4 V5) (5 T5 t5 V5 V6) (6 T6 t6 V6 Complete) Complete)
((0 T0 t0 V1) (1 T1 t1 V1 V2) (2 T2 t2 V2 V3) (3 T3 t3 V3 V4) (4 T4 t4 V4 V5) (5 T5 t5 V5 V6) (6 T6 t6 V6 V7) (7 T7 t7 V7 Complete) Complete)
((0 T0 t0 V1) (1 T1 t1 V1 V2) (2 T2 t2 V2 V3) (3 T3 t3 V3 V4) (4 T4 t4 V4 V5) (5 T5 t5 V5 V6) (6 T6 t6 V6 V7) (7 T7 t7 V7 V8) (8 T8 t8 V8 Complete) Complete)
((0 T0 t0 V1) (1 T1 t1 V1 V2) (2 T2 t2 V2 V3) (3 T3 t3 V3 V4) (4 T4 t4 V4 V5) (5 T5 t5 V5 V6) (6 T6 t6 V6 V7) (7 T7 t7 V7 V8) (8 T8 t8 V8 V9) (9 T9 t9 V9 Complete) Complete)
((0 T0 t0 V1) (1 T1 t1 V1 V2) (2 T2 t2 V2 V3) (3 T3 t3 V3 V4) (4 T4 t4 V4 V5) (5 T5 t5 V5 V6) (6 T6 t6 V6 V7) (7 T7 t7 V7 V8) (8 T8 t8 V8 V9) (9 T9 t9 V9 V10) (10 T10 t10 V10 Complete) Complete)

((0 T0 t0 Incomplete) Incomplete)
((0 T0 t0 V1) (1 T1 t1 V1 Incomplete) Incomplete)
((0 T0 t0 V1) (1 T1 t1 V1 V2) (2 T2 t2 V2 Incomplete) Incomplete)
((0 T0 t0 V1) (1 T1 t1 V1 V2) (2 T2 t2 V2 V3) (3 T3 t3 V3 Incomplete) Incomplete)
((0 T0 t0 V1) (1 T1 t1 V1 V2) (2 T2 t2 V2 V3) (3 T3 t3 V3 V4) (4 T4 t4 V4 Incomplete) Incomplete)
((0 T0 t0 V1) (1 T1 t1 V1 V2) (2 T2 t2 V2 V3) (3 T3 t3 V3 V4) (4 T4 t4 V4 V5) (5 T5 t5 V5 Incomplete) Incomplete)
((0 T0 t0 V1) (1 T1 t1 V1 V2) (2 T2 t2 V2 V3) (3 T3 t3 V3 V4) (4 T4 t4 V4 V5) (5 T5 t5 V5 V6) (6 T6 t6 V6 Incomplete) Incomplete)
((0 T0 t0 V1) (1 T1 t1 V1 V2) (2 T2 t2 V2 V3) (3 T3 t3 V3 V4) (4 T4 t4 V4 V5) (5 T5 t5 V5 V6) (6 T6 t6 V6 V7) (7 T7 t7 V7 Incomplete) Incomplete)
((0 T0 t0 V1) (1 T1 t1 V1 V2) (2 T2 t2 V2 V3) (3 T3 t3 V3 V4) (4 T4 t4 V4 V5) (5 T5 t5 V5 V6) (6 T6 t6 V6 V7) (7 T7 t7 V7 V8) (8 T8 t8 V8 Incomplete) Incomplete)
((0 T0 t0 V1) (1 T1 t1 V1 V2) (2 T2 t2 V2 V3) (3 T3 t3 V3 V4) (4 T4 t4 V4 V5) (5 T5 t5 V5 V6) (6 T6 t6 V6 V7) (7 T7 t7 V7 V8) (8 T8 t8 V8 V9) (9 T9 t9 V9 Incomplete) Incomplete)
((0 T0 t0 V1) (1 T1 t1 V1 V2) (2 T2 t2 V2 V3) (3 T3 t3 V3 V4) (4 T4 t4 V4 V5) (5 T5 t5 V5 V6) (6 T6 t6 V6 V7) (7 T7 t7 V7 V8) (8 T8 t8 V8 V9) (9 T9 t9 V9 V10) (10 T10 t10 V10 Incomplete) Incomplete)
}

const MAX_SZ: usize = 8;

macro_rules! sign_bit {
    ($type:ident) => {
        (1 << (mem::size_of::<$type>() * 8 - 1))
    };
}

macro_rules! unpack_ux {
    ($ux: ident, $input: expr, $n: expr) => {{
        let (input, bytes) = parse_bytes($input, $n)?;
        let mut arr = [0u8; ::std::mem::size_of::<$ux>()];
        (&mut arr[(::std::mem::size_of::<$ux>() - $n)..]).copy_from_slice(bytes);
        (input, $ux::from_be_bytes(arr))
    }};
}

macro_rules! unpack_px {
    ($ix: ident, $ux: ident, $input: expr, $n: expr) => {{
        let (input, bytes) = parse_bytes($input, $n)?;
        let mut arr = [0u8; ::std::mem::size_of::<$ux>()];
        (&mut arr[(::std::mem::size_of::<$ux>() - $n)..]).copy_from_slice(bytes);
        let x = $ix::from_be_bytes(arr);
        if x < 0 {
            Err(PackError::UnsupportedIntLength)
        } else {
            Ok((input, x))
        }
    }};
}
macro_rules! unpack_nx {
    ($ix: ident, $ux: ident, $input: expr, $n: expr) => {{
        let (input, bytes) = parse_bytes($input, $n)?;
        let mut arr = [0xffu8; ::std::mem::size_of::<$ix>()];
        (&mut arr[(::std::mem::size_of::<$ix>() - $n)..]).copy_from_slice(bytes);
        let x = $ix::from_be_bytes(arr).wrapping_add(1);
        if x > 0 {
            Err(PackError::UnsupportedIntLength)
        } else {
            Ok((input, x))
        }
    }};
}

macro_rules! impl_ux {
    ($ux: ident) => {
        impl_ux!($ux, mem::size_of::<$ux>());
    };
    ($ux: ident, $max_sz:expr) => {
        impl TuplePack<Complete, Complete> for $ux {
            fn pack<W: io::Write>(
                &self,
                mut w: PackedTuple<W, Complete>,
                _tuple_depth: TupleDepth,
            ) -> io::Result<PackedTuple<W, Complete>> {
                const SZ: usize = mem::size_of::<$ux>();
                let u = *self;
                let n = SZ - (u.leading_zeros() as usize) / 8;
                if SZ <= MAX_SZ || n <= MAX_SZ {
                    w.write_all(&[INTZERO + n as u8])?;
                } else {
                    w.write_all(&[POSINTEND, n as u8])?;
                };
                w.write_all(&u.to_be_bytes()[SZ - n..])?;
                Ok(w)
            }
        }

        impl<'de> TupleUnpack<'de> for $ux {
            fn unpack(input: &[u8], _tuple_depth: TupleDepth) -> PackResult<(&[u8], Self)> {
                const SZ: usize = mem::size_of::<$ux>();
                let (input, found) = parse_byte(input)?;
                if INTZERO <= found && found <= INTZERO + $max_sz as u8 {
                    let n = (found - INTZERO) as usize;
                    Ok(unpack_ux!($ux, input, n))
                } else if found == POSINTEND {
                    let (input, raw_length) = parse_byte(input)?;
                    let n: usize = usize::from(raw_length);
                    if n > SZ {
                        return Err(PackError::UnsupportedIntLength);
                    }
                    Ok(unpack_ux!($ux, input, n))
                } else {
                    Err(PackError::BadCode {
                        found,
                        expected: None,
                    })
                }
            }
        }
    };
}

macro_rules! impl_ix {
    ($ix: ident, $ux: ident) => {
        impl_ix!($ix, $ux, mem::size_of::<$ix>());
    };
    ($ix: ident, $ux: ident, $max_sz:expr) => {
        impl TuplePack<Complete, Complete> for $ix {
            fn pack<W: io::Write>(
                &self,
                mut w: PackedTuple<W, Complete>,
                _tuple_depth: TupleDepth,
            ) -> io::Result<PackedTuple<W, Complete>> {
                const SZ: usize = mem::size_of::<$ix>();
                let i = *self;
                let u = self.wrapping_abs() as $ux;
                let n = SZ - (u.leading_zeros() as usize) / 8;
                let arr = if i >= 0 {
                    if SZ <= MAX_SZ || n <= MAX_SZ {
                        w.write_all(&[INTZERO + n as u8])?;
                    } else {
                        w.write_all(&[POSINTEND, n as u8])?;
                    }
                    (u.to_be_bytes())
                } else {
                    if SZ <= MAX_SZ || n <= MAX_SZ {
                        w.write_all(&[INTZERO - n as u8])?;
                    } else {
                        w.write_all(&[NEGINTSTART, n as u8 ^ 0xff])?;
                    }
                    (i.wrapping_sub(1).to_be_bytes())
                };
                w.write_all(&arr[SZ - n..])?;

                Ok(w)
            }
        }

        impl<'de> TupleUnpack<'de> for $ix {
            fn unpack(input: &[u8], _tuple_depth: TupleDepth) -> PackResult<(&[u8], Self)> {
                const SZ: usize = mem::size_of::<$ix>();
                let (input, found) = parse_byte(input)?;
                if INTZERO <= found && found <= INTZERO + $max_sz as u8 {
                    let n = (found - INTZERO) as usize;
                    unpack_px!($ix, $ux, input, n)
                } else if INTZERO - $max_sz as u8 <= found && found < INTZERO {
                    let n = (INTZERO - found) as usize;
                    unpack_nx!($ix, $ux, input, n)
                } else if found == NEGINTSTART {
                    let (input, raw_length) = parse_byte(input)?;
                    let n = usize::from(raw_length ^ 0xff);
                    if n > SZ {
                        return Err(PackError::UnsupportedIntLength);
                    }
                    unpack_nx!($ix, $ux, input, n)
                } else if found == POSINTEND {
                    let (input, raw_length) = parse_byte(input)?;
                    let n: usize = usize::from(raw_length);
                    if n > SZ {
                        return Err(PackError::UnsupportedIntLength);
                    }
                    unpack_px!($ix, $ux, input, n)
                } else {
                    Err(PackError::BadCode {
                        found,
                        expected: None,
                    })
                }
            }
        }
    };
}

macro_rules! impl_fx {
    ( $fx: ident, $fx_to_ux_be_bytes: ident, $ux_width: tt, $parse_ux: ident, $ux: ident, $code: ident) => {
        #[inline]
        pub(super) fn $fx_to_ux_be_bytes(f: $fx) -> [u8; $ux_width] {
            let u = if f.is_sign_negative() {
                f.to_bits() ^ ::std::$ux::MAX
            } else {
                f.to_bits() ^ sign_bit!($ux)
            };
            u.to_be_bytes()
        }
        impl TuplePack<Complete, Complete> for $fx {
            fn pack<W: io::Write>(
                &self,
                mut w: PackedTuple<W, Complete>,
                _tuple_depth: TupleDepth,
            ) -> io::Result<PackedTuple<W, Complete>> {
                let bytes = $fx_to_ux_be_bytes(*self);
                w.write_all(&[$code])?;
                w.write_all(&bytes)?;
                Ok(w)
            }
        }

        fn $parse_ux(input: &[u8]) -> PackResult<(&[u8], $ux)> {
            let (input, bytes) = parse_bytes(input, mem::size_of::<$ux>())?;
            let mut arr = [0u8; mem::size_of::<$ux>()];
            arr.copy_from_slice(bytes);
            Ok((input, $ux::from_be_bytes(arr)))
        }
        impl<'de> TupleUnpack<'de> for $fx {
            fn unpack(input: &[u8], _tuple_depth: TupleDepth) -> PackResult<(&[u8], Self)> {
                let input = parse_code(input, $code)?;
                let (input, u) = $parse_ux(input)?;
                Ok((
                    input,
                    $fx::from_bits(if (u & sign_bit!($ux)) == 0 {
                        u ^ ::std::$ux::MAX
                    } else {
                        u ^ sign_bit!($ux)
                    }),
                ))
            }
        }
    };
}

//impl_ux!(u8);
impl_ux!(u16);
impl_ux!(u32);
impl_ux!(u64);
impl_ux!(u128, MAX_SZ);
impl_ux!(usize);

//impl_ix!(i8, u8);
impl_ix!(i16, u16);
impl_ix!(i32, u32);
impl_ix!(i64, u64);
impl_ix!(i128, u128, MAX_SZ);
impl_ix!(isize, usize);

impl_fx!(f32, f32_to_u32_be_bytes, 4, parse_u32, u32, FLOAT);
impl_fx!(f64, f64_to_u64_be_bytes, 8, parse_u64, u64, DOUBLE);

#[cfg(feature = "num-bigint")]
mod bigint {
    use super::*;
    use num_bigint::{BigInt, BigUint, Sign};
    use std::convert::TryFrom;

    fn invert(bytes: &mut [u8]) {
        // The ones' complement of a binary number is defined as the value
        // obtained by inverting all the bits in the binary representation
        // of the number (swapping 0s for 1s and vice versa).
        for byte in bytes.iter_mut() {
            *byte = !*byte;
        }
    }

    fn inverted(bytes: &[u8]) -> Vec<u8> {
        // The ones' complement of a binary number is defined as the value
        // obtained by inverting all the bits in the binary representation
        // of the number (swapping 0s for 1s and vice versa).
        bytes.iter().map(|byte| !*byte).collect()
    }

    fn bigint_n(n: usize) -> io::Result<u8> {
        u8::try_from(n).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "BigUint requires more than 255 bytes to be represented",
            )
        })
    }

    impl TuplePack<Complete, Complete> for BigInt {
        fn pack<W: io::Write>(
            &self,
            mut w: PackedTuple<W, Complete>,
            _tuple_depth: TupleDepth,
        ) -> io::Result<PackedTuple<W, Complete>> {
            if self.sign() == Sign::NoSign {
                w.write_all(&[INTZERO])?;
                return Ok(w);
            }
            let (sign, mut bytes) = self.to_bytes_be();
            let n = bytes.len();
            match sign {
                Sign::Minus => {
                    if n <= MAX_SZ {
                        w.write_all(&[INTZERO - n as u8])?;
                    } else {
                        w.write_all(&[NEGINTSTART, bigint_n(n)? ^ 0xff])?;
                    }
                    invert(&mut bytes);
                    w.write_all(&bytes)?;
                }
                Sign::NoSign => unreachable!(),
                Sign::Plus => {
                    if n <= MAX_SZ {
                        w.write_all(&[INTZERO + n as u8])?;
                    } else {
                        w.write_all(&[POSINTEND, bigint_n(n)?])?;
                    }
                    w.write_all(&bytes)?;
                }
            }

            Ok(w)
        }
    }

    impl<'de> TupleUnpack<'de> for BigInt {
        fn unpack(input: &[u8], _tuple_depth: TupleDepth) -> PackResult<(&[u8], Self)> {
            let (input, found) = parse_byte(input)?;
            if INTZERO <= found && found <= INTZERO + MAX_SZ as u8 {
                let n = (found - INTZERO) as usize;
                let (input, bytes) = parse_bytes(input, n)?;
                Ok((input, Self::from_bytes_be(Sign::Plus, bytes)))
            } else if INTZERO - MAX_SZ as u8 <= found && found < INTZERO {
                let n = (INTZERO - found) as usize;
                let (input, bytes) = parse_bytes(input, n)?;
                Ok((input, Self::from_bytes_be(Sign::Minus, &inverted(bytes))))
            } else if found == NEGINTSTART {
                let (input, raw_length) = parse_byte(input)?;
                let n = usize::from(raw_length ^ 0xff);
                let (input, bytes) = parse_bytes(input, n)?;
                Ok((input, Self::from_bytes_be(Sign::Minus, &inverted(bytes))))
            } else if found == POSINTEND {
                let (input, raw_length) = parse_byte(input)?;
                let n: usize = usize::from(raw_length);
                let (input, bytes) = parse_bytes(input, n)?;
                Ok((input, Self::from_bytes_be(Sign::Plus, bytes)))
            } else {
                Err(PackError::BadCode {
                    found,
                    expected: None,
                })
            }
        }
    }

    impl TuplePack<Complete, Complete> for BigUint {
        fn pack<W: io::Write>(
            &self,
            mut w: PackedTuple<W, Complete>,
            _tuple_depth: TupleDepth,
        ) -> io::Result<PackedTuple<W, Complete>> {
            let n = self.bits();
            if n == 0 {
                w.write_all(&[INTZERO])?;
                return Ok(w);
            }
            let bytes = self.to_bytes_be();
            let n = bytes.len();
            if n <= MAX_SZ {
                w.write_all(&[INTZERO + n as u8])?;
            } else {
                w.write_all(&[POSINTEND, bigint_n(n)?])?;
            }
            w.write_all(&bytes)?;

            Ok(w)
        }
    }

    impl<'de> TupleUnpack<'de> for BigUint {
        fn unpack(input: &[u8], _tuple_depth: TupleDepth) -> PackResult<(&[u8], Self)> {
            let (input, found) = parse_byte(input)?;
            if INTZERO <= found && found <= INTZERO + MAX_SZ as u8 {
                let n = (found - INTZERO) as usize;
                let (input, bytes) = parse_bytes(input, n)?;
                Ok((input, Self::from_bytes_be(bytes)))
            } else if found == POSINTEND {
                let (input, raw_length) = parse_byte(input)?;
                let n: usize = usize::from(raw_length);
                let (input, bytes) = parse_bytes(input, n)?;
                Ok((input, Self::from_bytes_be(bytes)))
            } else {
                Err(PackError::BadCode {
                    found,
                    expected: None,
                })
            }
        }
    }
}

impl TuplePack<Complete, Complete> for bool {
    fn pack<W: io::Write>(
        &self,
        mut w: PackedTuple<W, Complete>,
        _tuple_depth: TupleDepth,
    ) -> io::Result<PackedTuple<W, Complete>> {
        w.write_all(&[if *self { TRUE } else { FALSE }])?;
        Ok(w)
    }
}

impl<'de> TupleUnpack<'de> for bool {
    fn unpack(input: &[u8], _tuple_depth: TupleDepth) -> PackResult<(&[u8], Self)> {
        let (input, v) = parse_byte(input)?;
        match v {
            FALSE => Ok((input, false)),
            TRUE => Ok((input, true)),
            _ => Err(PackError::Message(
                format!("{} is not a valid bool value", v).into_boxed_str(),
            )),
        }
    }
}

impl<'b, T> TuplePack<Complete, Complete> for &'b [T]
where
    T: TuplePack<Complete, Complete>,
{
    fn pack<W: io::Write>(
        &self,
        mut w: PackedTuple<W, Complete>,
        tuple_depth: TupleDepth,
    ) -> io::Result<PackedTuple<W, Complete>> {
        if tuple_depth.depth() > 0 {
            w.write_all(&[NESTED])?;
        }

        for v in self.iter() {
            w = v.pack(w, tuple_depth.increment())?;
        }

        if tuple_depth.depth() > 0 {
            w.write_all(&[NIL])?;
        }
        Ok(w)
    }
}

impl<T> TuplePack<Complete, Complete> for Vec<T>
where
    T: TuplePack<Complete, Complete>,
{
    fn pack<W: io::Write>(
        &self,
        mut w: PackedTuple<W, Complete>,
        tuple_depth: TupleDepth,
    ) -> io::Result<PackedTuple<W, Complete>> {
        self.as_slice().pack(w, tuple_depth)
    }
}

fn is_end_of_tuple(input: &[u8], nested: bool) -> bool {
    match input.first() {
        None => true,
        _ if !nested => false,
        Some(&NIL) => Some(&ESCAPE) != input.get(1),
        _ => false,
    }
}

impl<'de, T> TupleUnpack<'de> for Vec<T>
where
    T: TupleUnpack<'de>,
{
    fn unpack(mut input: &'de [u8], tuple_depth: TupleDepth) -> PackResult<(&'de [u8], Self)> {
        let nested = tuple_depth.depth() > 0;
        if nested {
            input = parse_code(input, NESTED)?;
        }

        let mut vec = Vec::new();

        while !is_end_of_tuple(input, nested) {
            let (rem, v) = T::unpack(input, tuple_depth.increment())?;
            input = rem;
            vec.push(v);
        }

        if nested {
            input = parse_code(input, NIL)?;
        }

        Ok((input, vec))
    }
}

impl<'b> TuplePack<Complete, Complete> for Bytes<'b> {
    fn pack<W: io::Write>(
        &self,
        mut w: PackedTuple<W, Complete>,
        _tuple_depth: TupleDepth,
    ) -> io::Result<PackedTuple<W, Complete>> {
        w.write_all(&[BYTES])?;
        write_bytes(&mut w, self.as_ref())?;
        Ok(w)
    }
}

impl<'de> TupleUnpack<'de> for Bytes<'de> {
    fn unpack(input: &'de [u8], _tuple_depth: TupleDepth) -> PackResult<(&'de [u8], Self)> {
        let input = parse_code(input, BYTES)?;
        let (input, v) = parse_slice(input)?;
        Ok((input, Bytes(v)))
    }
}

impl<'b> TuplePack<Complete, Complete> for &'b [u8] {
    fn pack<W: io::Write>(
        &self,
        mut w: PackedTuple<W, Complete>,
        tuple_depth: TupleDepth,
    ) -> io::Result<PackedTuple<W, Complete>> {
        Bytes::from(*self).pack(w, tuple_depth)
    }
}

impl TuplePack<Complete, Complete> for Vec<u8> {
    fn pack<W: io::Write>(
        &self,
        w: PackedTuple<W, Complete>,
        tuple_depth: TupleDepth,
    ) -> io::Result<PackedTuple<W, Complete>> {
        Bytes::from(self.as_slice()).pack(w, tuple_depth)
    }
}

impl<'de> TupleUnpack<'de> for Vec<u8> {
    fn unpack(input: &'de [u8], tuple_depth: TupleDepth) -> PackResult<(&'de [u8], Self)> {
        let (input, bytes) = Bytes::unpack(input, tuple_depth)?;
        Ok((input, bytes.into_owned()))
    }
}

impl<'b> TuplePack<Complete, Complete> for &'b str {
    fn pack<W: io::Write>(
        &self,
        mut w: PackedTuple<W, Complete>,
        _tuple_depth: TupleDepth,
    ) -> io::Result<PackedTuple<W, Complete>> {
        w.write_all(&[STRING])?;
        write_bytes(&mut w, self.as_bytes())?;
        Ok(w)
    }
}

impl TuplePack<Complete, Complete> for String {
    fn pack<W: io::Write>(
        &self,
        w: PackedTuple<W, Complete>,
        tuple_depth: TupleDepth,
    ) -> io::Result<PackedTuple<W, Complete>> {
        self.as_str().pack(w, tuple_depth)
    }
}

impl<'b> TuplePack<Complete, Complete> for Cow<'b, str> {
    fn pack<W: io::Write>(
        &self,
        w: PackedTuple<W, Complete>,
        tuple_depth: TupleDepth,
    ) -> io::Result<PackedTuple<W, Complete>> {
        self.as_ref().pack(w, tuple_depth)
    }
}

impl<'de> TupleUnpack<'de> for Cow<'de, str> {
    fn unpack(input: &'de [u8], _tuple_depth: TupleDepth) -> PackResult<(&'de [u8], Self)> {
        let input = parse_code(input, STRING)?;
        let (input, v) = parse_string(input)?;
        Ok((input, v))
    }
}

impl<'de> TupleUnpack<'de> for String {
    fn unpack(input: &[u8], _tuple_depth: TupleDepth) -> PackResult<(&[u8], Self)> {
        let input = parse_code(input, STRING)?;
        let (input, v) = parse_string(input)?;
        Ok((input, v.into_owned()))
    }
}

impl<T> TuplePack<Complete, Complete> for Option<T>
where
    T: TuplePack<Complete, Complete>,
{
    fn pack<W: io::Write>(
        &self,
        mut w: PackedTuple<W, Complete>,
        tuple_depth: TupleDepth,
    ) -> io::Result<PackedTuple<W, Complete>> {
        match self {
            None => {
                if tuple_depth.depth() > 1 {
                    // Empty value in nested tuple is encoded with [NIL, ESCAPE] to
                    // disambiguate itself with end-of-tuple marker.
                    w.write_all(&[NIL, ESCAPE])?;
                } else {
                    w.write_all(&[NIL])?;
                }
                Ok(w)
            }
            Some(v) => v.pack(w, tuple_depth),
        }
    }
}

impl<'de, T> TupleUnpack<'de> for Option<T>
where
    T: TupleUnpack<'de>,
{
    fn unpack(mut input: &'de [u8], tuple_depth: TupleDepth) -> PackResult<(&'de [u8], Self)> {
        if let Some(&NIL) = input.first() {
            input = parse_code(input, NIL)?;
            if tuple_depth.depth() > 1 {
                input = parse_code(input, ESCAPE)?;
            }
            Ok((input, None))
        } else {
            let (input, v) = T::unpack(input, tuple_depth)?;
            Ok((input, Some(v)))
        }
    }
}

impl<'b> TuplePack<Complete, Complete> for Element<'b> {
    fn pack<W: io::Write>(
        &self,
        w: PackedTuple<W, Complete>,
        tuple_depth: TupleDepth,
    ) -> io::Result<PackedTuple<W, Complete>> {
        match self {
            Element::Nil => Option::<()>::None.pack(w, tuple_depth),
            Element::Bool(b) => b.pack(w, tuple_depth),
            Element::Int(i) => i.pack(w, tuple_depth),
            Element::Float(f) => f.pack(w, tuple_depth),
            Element::Double(f) => f.pack(w, tuple_depth),
            Element::String(ref c) => c.pack(w, tuple_depth),
            Element::Bytes(ref b) => b.pack(w, tuple_depth),
            Element::Versionstamp(ref b) if b.is_complete() => b.pack(w, tuple_depth),
            Element::Versionstamp(_) => {
                panic!("unsupported packing of incomplete versionstamp through elements")
            }
            Element::Tuple(ref v) => v.pack(w, tuple_depth),
            #[cfg(feature = "uuid")]
            Element::Uuid(v) => v.pack(w, tuple_depth),
            #[cfg(feature = "num-bigint")]
            Element::BigInt(v) => v.pack(w, tuple_depth),
        }
    }
}

impl<'de> TupleUnpack<'de> for Element<'de> {
    fn unpack(input: &'de [u8], tuple_depth: TupleDepth) -> PackResult<(&'de [u8], Self)> {
        const INTMIN: u8 = INTZERO - 8;
        const INTMAX: u8 = INTZERO + 8;

        let first = match input.first() {
            None if tuple_depth.depth() == 0 => return Ok((input, Element::Tuple(Vec::new()))),
            None => return Err(PackError::MissingBytes),
            Some(byte) => byte,
        };

        let (mut input, mut v) = match *first {
            NIL => {
                let (input, _) = Option::<()>::unpack(input, tuple_depth)?;
                (input, Element::Nil)
            }
            BYTES => {
                let (input, v) = Bytes::unpack(input, tuple_depth)?;
                (input, Element::Bytes(v))
            }
            STRING => {
                let (input, v) = Cow::<'de, str>::unpack(input, tuple_depth)?;
                (input, Element::String(v))
            }
            NESTED => {
                let (input, v) = Vec::<Self>::unpack(input, tuple_depth)?;
                (input, Element::Tuple(v))
            }
            INTMIN..=INTMAX => match i64::unpack(input, tuple_depth) {
                Ok((input, v)) => (input, Element::Int(v)),
                #[cfg(feature = "num-bigint")]
                Err(PackError::UnsupportedIntLength) => {
                    let (input, v) = num_bigint::BigInt::unpack(input, tuple_depth)?;
                    (input, Element::BigInt(v))
                }
                Err(err) => return Err(err),
            },
            #[cfg(feature = "num-bigint")]
            NEGINTSTART => {
                let (input, v) = num_bigint::BigInt::unpack(input, tuple_depth)?;
                (input, Element::BigInt(v))
            }
            #[cfg(feature = "num-bigint")]
            POSINTEND => {
                let (input, v) = num_bigint::BigInt::unpack(input, tuple_depth)?;
                (input, Element::BigInt(v))
            }
            #[cfg(not(feature = "num-bigint"))]
            NEGINTSTART => {
                let (input, v) = i64::unpack(input, tuple_depth)?;
                (input, Element::Int(v))
            }
            #[cfg(not(feature = "num-bigint"))]
            POSINTEND => {
                let (input, v) = i64::unpack(input, tuple_depth)?;
                (input, Element::Int(v))
            }
            FLOAT => {
                let (input, v) = f32::unpack(input, tuple_depth)?;
                (input, Element::Float(v))
            }
            DOUBLE => {
                let (input, v) = f64::unpack(input, tuple_depth)?;
                (input, Element::Double(v))
            }
            FALSE | TRUE => {
                let (input, v) = bool::unpack(input, tuple_depth)?;
                (input, Element::Bool(v))
            }
            VERSIONSTAMP => {
                let (input, v) = Versionstamp::unpack(input, tuple_depth)?;
                (input, Element::Versionstamp(v))
            }
            #[cfg(feature = "uuid")]
            UUID => {
                let (input, v) = uuid::Uuid::unpack(input, tuple_depth)?;
                (input, Element::Uuid(v))
            }
            found => {
                return Err(PackError::BadCode {
                    found,
                    expected: None,
                });
            }
        };

        if tuple_depth.depth() == 0 && !input.is_empty() {
            let mut tuple = vec![v];
            while !input.is_empty() {
                let (rem, v) = Self::unpack(input, tuple_depth.increment())?;
                tuple.push(v);
                input = rem;
            }
            v = Element::Tuple(tuple);
        }

        Ok((input, v))
    }
}

impl TuplePack<Complete, Complete> for Versionstamp {
    fn pack<W: io::Write>(
        &self,
        mut w: PackedTuple<W, Complete>,
        _tuple_depth: TupleDepth,
    ) -> io::Result<PackedTuple<W, Complete>> {
        if !self.is_complete() {
            panic!("cannot pack incomplete versionstamp. Use IncompleteVersionstamp instead");
        }
        w.write_all(&[VERSIONSTAMP])?;
        w.write_all(self.as_bytes())?;
        Ok(w)
    }
}

impl<'de> TupleUnpack<'de> for Versionstamp {
    fn unpack(input: &[u8], _tuple_depth: TupleDepth) -> PackResult<(&[u8], Self)> {
        let input = parse_code(input, VERSIONSTAMP)?;
        let (input, slice) = parse_bytes(input, 12)?;
        let mut bytes = [0xff; 12];
        bytes.copy_from_slice(slice);
        Ok((input, Versionstamp::from(bytes)))
    }
}
#[cfg(feature = "uuid")]
mod pack_uuid {
    use super::*;
    use uuid::Uuid;

    impl TuplePack<Complete, Complete> for Uuid {
        fn pack<W: io::Write>(
            &self,
            mut w: PackedTuple<W, Complete>,
            _tuple_depth: TupleDepth,
        ) -> io::Result<PackedTuple<W, Complete>> {
            w.write_all(&[UUID])?;
            w.write_all(self.as_bytes())?;
            Ok(w)
        }
    }

    impl<'de> TupleUnpack<'de> for Uuid {
        fn unpack(input: &[u8], _tuple_depth: TupleDepth) -> PackResult<(&[u8], Self)> {
            let input = parse_code(input, UUID)?;
            let (input, slice) = parse_bytes(input, 16)?;
            let uuid = Self::from_slice(slice).map_err(|_| PackError::BadUuid)?;
            Ok((input, uuid))
        }
    }
}
