#pragma once
namespace cutlass::epilogue::threadblock {
using namespace cute;
using namespace detail;
template<
  class ThreadMap,
  class Element,
  class StrideMNL
>
struct VisitorIndexedGateLoad{

  struct Arguments {
    Element* ptr_aux = nullptr;
    Element null_default = Element(0);
    StrideMNL dAux = {};
    int32_t const* rows = nullptr;
    int n = 0;
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  // Software pipeline stages
  static const int Stages = ThreadMap::Stages;

  struct SharedStorage {};

  // Global load type
  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE
  VisitorIndexedGateLoad() { }

  CUTLASS_HOST_DEVICE
  VisitorIndexedGateLoad(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }

  Params const* params_ptr;

  template <class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
      GTensor&& tC_gAux,
      RTensor&& tC_rAux,
      CTensor&& tC_cAux,
      ProblemShape problem_shape,
      Params const* params_ptr
    ):
      tC_gAux(cute::forward<GTensor>(tC_gAux)),
      tC_rAux(cute::forward<RTensor>(tC_rAux)),
      tC_cAux(cute::forward<CTensor>(tC_cAux)),
      problem_shape(problem_shape),
      params_ptr(params_ptr) { }

    GTensor tC_gAux;
    RTensor tC_rAux;
    CTensor tC_cAux;
    Params const* params_ptr;
    ProblemShape problem_shape;

    CUTLASS_DEVICE void
    begin_step(int step_idx) {
      clear(tC_rAux(_,_,_,step_idx%Stages));
      auto src_v = filter(tC_gAux(_,_,_,step_idx));
      auto coord_v = filter(tC_cAux(_,_,_,step_idx));
      auto dst_v = filter(tC_rAux(_,_,_,step_idx%Stages));
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(src_v); ++i) {
        bool guard = elem_less(coord_v(i), problem_shape);
        auto coord=coord_v(i);
        Element const* addr=params_ptr->ptr_aux;
        if(guard) addr += int64_t(params_ptr->rows[get<0>(coord)])*params_ptr->n+get<1>(coord);
        cutlass::arch::global_load<VecType, sizeof(VecType)>(dst_v(i), (void const*)addr, guard);
      }
    }

    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      Tensor tC_rAux_frg = recast<Array<Element, FragmentSize>>(coalesce(tC_rAux(_,_,_,iter_idx%Stages)));
      return tC_rAux_frg(frg_idx);
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    Tensor mAux = make_tensor(
      make_gmem_ptr(params_ptr->ptr_aux),
      problem_shape,
      params_ptr->dAux);   // (M,N,L)
    // VECTOR, FRAGMENT_COLUMN, FRAGMENT_ROW, ITERATION_ROW, ITERATION_GROUP, ITERATION_CLUSTER
    Tensor tC_gAux = recast<VecType>(
      group_modes<3,6>(ThreadMap::partition(mAux, thread_idx, threadblock_tile_offset)));
    // VECTOR, FRAGMENT_COLUMN, FRAGMENT_ROW, Stages
    Tensor tC_rAux = make_tensor<VecType>(
      make_layout(flatten(make_shape(take<0,3>(tC_gAux.shape()), Int<Stages>{}))));

    // Generate the pred tensor
    Tensor cAux = make_identity_tensor(mAux.shape());
    Tensor tC_cAux = outer_partition(
      group_modes<3,6>(ThreadMap::partition(cAux, thread_idx, threadblock_tile_offset)),
      Shape<Int<VecLength>>{},
      (_0{})
    );

    return Callbacks<
      decltype(tC_gAux), decltype(tC_rAux),
      decltype(tC_cAux), ProblemShape>(
      cute::move(tC_gAux),
      cute::move(tC_rAux),
      cute::move(tC_cAux),
      problem_shape,
      params_ptr
    );
  }
};

}
