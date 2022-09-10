pub trait PdmBuffer<const LEN: usize> {
    fn as_ref(&self) -> &[u8; LEN]; 
    fn as_mut(&mut self) -> &mut [u8; LEN];

    fn ptr(&mut self) -> *mut [u8; LEN];
}

pub struct PdmBufferPtr<const LEN: usize> {
    buf: *mut [u8; LEN],
}

impl<const LEN: usize> PdmBuffer<LEN> for PdmBufferPtr<LEN>
{
    fn as_ref(&self) -> &[u8; LEN] {
        unsafe { &*self.buf }
    }
    fn as_mut(&mut self) -> &mut [u8; LEN] {
        unsafe { &mut *self.buf }
    }

    fn ptr(&mut self) -> *mut [u8; LEN] {
        self.buf
    }
}

pub struct PdmBufferHeap<const LEN: usize> {
    buf: [u8; LEN],
}

impl<const LEN: usize> PdmBuffer<LEN> for PdmBufferHeap<LEN>
{
    fn as_ref(&self) -> &[u8; LEN] {
        &self.buf
    }
    fn as_mut(&mut self) -> &mut [u8; LEN] {
        &mut self.buf
    }

    fn ptr(&mut self) -> *mut [u8; LEN] {
        self.buf.as_mut_ptr() as *mut [u8; LEN]
    }
}
