/*!
 *  Copyright (c) 2015 by Contributors
 * \file operator.h
 * \brief Operator interface of mxnet.
 * \author Naiyan Wang
 */
#ifndef MXNET_OPERATOR_H_
#define MXNET_OPERATOR_H_

#include <dmlc/base.h>
#include <dmlc/json.h>
#include <dmlc/logging.h>
#include <dmlc/registry.h>
#include <vector>
#include <string>
#include <utility>
#include "./base.h"
#include "./resource.h"

namespace mxnet {
/*! \brief operation request type to Forward and Backward */
enum OpReqType {
  /*! \brief no operation, do not write anything */
  kNullOp,
  /*! \brief write gradient to provided space */
  kWriteTo,
  /*!
   * \brief perform an inplace write,
   * Target shares memory with one of input arguments.
   * This option only happen when
   */
  kWriteInplace,
  /*! \brief add to the provided space */
  kAddTo
};

/*!
 * \brief All the possible information needed by Operator.Forward and Backward
 *  This is the superset of RunContext.
 *  We use this data structure to bookkeep everything needed by Forward and Backward.
 * \sa Resource
 */
struct OpContext {
  /*! \brief whether it is training phase */
  int is_train;
  /*! \brief RunContext related resources */
  RunContext run_ctx;
  /*! \brief Resources requested by the operator */
  std::vector<Resource> requested;
  /*!
   * \brief get mshadow stream from Context
   * \return the mshadow stream
   * \tparam xpu the device type of the stream
   */
  template<typename xpu>
  inline mshadow::Stream<xpu>* get_stream() const {
    return run_ctx.get_stream<xpu>();
  }
};

/*!
 * \brief Operator interface.
 *  Operator defins basic operation unit of optimized computation graph in mxnet.
 *  This interface relies on pre-allocated memory in TBlob, the caller need to set
 *  the memory region in TBlob correctly before calling Forward and Backward.
 *
 *  Operator is generated by OperatorProperty.
 *  To add new operator(aka. layers of neural nets) to mxnet, developer need to create
 *  a new OperatorProperty and its corresponding Operator.
 *
 * \sa TBlob, TShape, OperatorProperty
 */
class Operator {
 public:
  /*! \brief destructor */
  virtual ~Operator() {}
  /*!
   * \brief perform a forward operation of Operator, save the output to TBlob.
   * \param ctx runtime context available to this call
   * \param in_data array of input data, it is const
   * \param req the request types of saving operation, can only be kWriteTo or kWriteInplace.
   * \param out_data array of output data, pointer is used to indicate that this is holder
   *        the space of TBlob in out_data must be pre-allocated with InferShape
   * \param aux_states Auxiliary states of operator. Normally operator doesn't
   *        need, epecial case like Batch Norm requires.
   * \sa OpReqType, OpContext
   */
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) = 0;
  /*!
   * \brief Perform a Backward Operation, write gradient to the in_grad.
   *
   * \note
   * Convention:
   *   out_grad.size() == OperatorProperty.NumVisibleOutputs()
   *   out_data.size() == OperatorProperty.NumOutputs()
   * out_data can contain additional invisible returns that remembers the
   * state carried from the Forward pass. For example mask in the dropout.
   * The gradients are passed from visible returns in this function.
   *
   * \par
   * Not all the TBlobs in the arguments will be available
   * if you override the DeclareBackwardDependency of corresponding OperatorProperty class.
   * Only the dependencies you declared will be available at corresponding position,
   * the rest of the parameters are simply dummy where you will get a nullptr.
   * You will be safe if you use the default DeclareBackwardDependency.
   * But only declare what you need will give engine more chance for optimization.
   *
   * \param ctx runtime context available to this call
   * \param out_grad the gradient value we get from of the Operator.
   * \param in_data the array of input data.
   * \param out_data the array of output data.
   * \param req request types of the saving operation, can be all types.
   * \param in_grad the array of gradient we need to write to.
   * \param aux_states Auxiliary states of operator. Normally operator doesn't need
   * \sa OperatorProperty, OpReqType, OpContext
   */
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    LOG(FATAL) << "Backward is not implemented";
  }
};

#if DMLC_USE_CXX11
// OperatorProperty allows C++11, while Operator do not rely on it.
/*!
 * \brief OperatorProperty is a object that stores all information about Operator.
 * It also contains method to generate context(device) specific operators.
 *
 * It also contains various functions that can be optimally overriden to
 * provide optimization chance for computation engine.
 */
class OperatorProperty {
 public:
  /*!
   * \brief virtual destructor
   */
  virtual ~OperatorProperty() {}
  /*!
   *  \brief Initialize the Operator by setting the parameters
   *  This function need to be called before all other functions.
   *  \param kwargs the keyword arguments parameters
   */
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) = 0;
  /*!
   * \brief Get input arguments of the Operator.
   * \return vector of arguments.
   */
  virtual std::vector<std::string> ListArguments() const {
    return {"data"};
  }
  /*!
   * \brief Get name of output values of Operator
   * \return name of output values.
   */
  virtual std::vector<std::string> ListOutputs() const {
    return {"output"};
  }
  /*!
   * \brief Get name of auxilary states of Operator
   * \return name of return values.
   */
  virtual std::vector<std::string> ListAuxiliaryStates() const {
    return {};
  }
  /*! \return number of real return values of the Operator */
  virtual int NumOutputs() const {
    return 1;
  }
  /*!
   * \brief get number of visible return values during Symbol creation.
   *  If NumVisibleOutputs() = k, and NumOutputs() = n.
   *  The first k returns will be presented in the resulting symbol.
   *
   *  The rest of the returns can be used for auxiliary states for Backward.
   *  For example, Dropout will return [data, mask], with NumVisibleOutputs() == 1.
   *  So when user call sym = Dropout(input), only data is presented in sym.
   *  But all the returns will be presented in out_data parameter of Backward if requested.
   *
   * \return number of default return values
   */
  virtual int NumVisibleOutputs() const {
    return NumOutputs();
  }
  /*!
   * \brief infer the shapes of outputs and unknown input arguments
   * \param in_shape the shape of input arguments of the operator
   *     this should be of same length as the vector returned by DescribeArgs
   *     in_shape allows unknown elements, which are checked by shape.ndim() == 0.
   *     For unknown shapes, InferShape will try to fill in the correct Shape in in_shape
   *     For known shapes, InferShape will check shape consistency
   *
   *     common practice: set the shape of data input, and usually weight's shape can be infered
   *
   * \param out_shape the shape of outputs of the operator
   *     InferShape will modify the vector to fill output TShape
   * \param aux_shape the shape of auxiliary states of the operator
   *     InferShape will modify the vector to fill output TShape
   * \return true if the shape inference is successful, false if there is not enough information.
   * \throws dmlc::Error if the known arg_shapes are inconsistent.
   */
  virtual bool InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape,
                          std::vector<TShape> *aux_shape) const = 0;
  /*!
   * \brief Copy this OperatorProperty.
   * \return a pointer to the copied OperatorProperty
   */
  virtual OperatorProperty* Copy() const = 0;
  /*!
   * \brief Create a Operator on specific context
   */
  virtual Operator* CreateOperator(Context ctx) const = 0;
  /*!
   * \brief return the type string of the Operator
   *  subclasses override this function.
   */
  virtual std::string TypeString() const = 0;
  //--------------------------------------------------------
  // All the below functions are optional to override.
  //--------------------------------------------------------
  /*!
   * \brief Declare additional resource required in forward pass.
   *  These additional resources will be presented in OpContext.requested
   *  in the same order of the returned Resource.
   * \param in_shape The input shape to the operator, corresponds to shapes of in_data.
   * \return Additional resource request
   */
  virtual std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const {
    return std::vector<ResourceRequest>();
  }
  /*!
   * \brief Decalre additional resource required in backward pass.
   *  These additional resources will be presented in OpContext.requested
   *  in the same order of the returned Resource.
   * \param in_shape The input shape to the operator, corresponds to shapes of in_data.
   * \return Additional resource request
   */
  virtual std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const {
    return std::vector<ResourceRequest>();
  }
  /*!
   * \brief Declare the input requirement of Backward pass.
   *
   *  Only the returned list of variables will be used in Backward.
   *  This function is used for memory optimization.
   *  It is adviced to override and only return what is actually needed.
   *  If this function is not overriden, all the variables will be valid in Backward.
   *
   * \code
   *  // The following code declares Backward need out_grad[0], in_data[0],in_data[1]
   *  vector<int> BackwardInputs(const vector<int> &out_grad,
   *                             const vector<int> &in_data,
   *                             const vector<int> &out_data) const {
   *    return {out_grad[0], in_data[0], in_data[1]};
   *  }
   * \endcode
   * \param out_grad gradient of outputs in backward pass.
   * \param in_data the input data in forward pass.
   * \param out_data the output data in forward pass.
   * \return an integer vector indicating the input requirments
   * \sa BackwardInputs
   */
  virtual std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const {
    // By default requires to see all the things.
    // remember to override this function to get a better performance.
    std::vector<int> ret = out_grad;
    ret.insert(ret.end(), in_data.begin(), in_data.end());
    ret.insert(ret.end(), out_data.begin(), out_data.end());
    return ret;
  }
  /*!
   * \brief Get possible forward inplace options.
   *  This function enables optimization to reuse memory of inputs in output.
   *  Only override when necessary, by default in-place is disabled.
   *
   *  The reason for void* type in the out_data is to distinguish the order
   *  of mappings between the two, compiler will report error when
   *  in_data and out_data's order in the pair get reversed.
   *
   * \code
   *  // The following code says out_data[0] can share data with in_data[0]
   *  vector<pair<int, void*> > ForwardInplaceOption(const vector<int> &in_data,
   *                                                 const vector<void*> &out_data) const {
   *    return {{in_data[0], out_data[0]}};
   *  }
   * \endcode
   * \param in_data The input data in forward pass.
   * \param out_data The output data in forward pass.
   * \return list of pair of that maps input->output,
   *   indicating possible in place operations.
   */
  virtual std::vector<std::pair<int, void*> > ForwardInplaceOption(
      const std::vector<int> &in_data,
      const std::vector<void*> &out_data) const {
    return std::vector<std::pair<int, void*> >();
  }
  /*!
   * \brief Get possible backward inplace options.
   *  This function enables optimization to reuse memory of inputs in output.
   *  Only override when necessary, by default in-place is disabled.
   *
   *  The reason for void* type in the in_grad is to distinguish the order
   *  of mappings between the two, compiler will report error when
   *  in_data and out_data's order in the pair get reversed.
   *
   * \code
   *  // The following code says in_grad[0] can share data with in_data[0]
   *  vector<pair<int,int> > BackwardInplaceOption(
   *                 const std::vector<int> &out_grad,
   *                 const std::vector<int> &in_data,
   *                 const std::vector<int> &out_data,
   *                 const std::vector<int> &in_grad) const {
   *    return {in_data[0], in_grad[0]}};
   *  }
   * \endcode
   * \param in_data The input data in forward pass.
   * \param out_data The output data in forward pass.
   * \param in_grad Gradient of inputs in backward pass.
   * \param out_grad Gradient of outputs in backward pass.
   * \return list of pair of that maps input->output,
   *   indicating possible in place operations.
   */
  virtual std::vector<std::pair<int, void*> > BackwardInplaceOption(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data,
      const std::vector<void*> &in_grad) const {
    return std::vector<std::pair<int, void*> >();
  }
  /*!
   * \brief Get Backward Input Dependency for generic types of data.
   *  Normally T can be pointer of Symbol::DataEntry, or NDArray.
   *  This function will select the result list of T according to DeclareBackwardDependency.
   *
   * \param in_data the input data in forward pass.
   * \param out_data the output data in forward pass.
   * \param out_grad gradient of outputs in backward pass.
   * \tparam T the generic type parameter.
   * \return vector of inputs the Backward Operation depends on.
   * \sa DeclareBackwardDependency
   */
  template<typename T>
  inline std::vector<T> BackwardInputs(const std::vector<T> &out_grad,
                                       const std::vector<T> &in_data,
                                       const std::vector<T> &out_data) const {
    int counter = 0;
    std::vector<int> out_grad_index(out_grad.size());
    std::vector<int> in_data_index(in_data.size());
    std::vector<int> out_data_index(out_data.size());
    for (size_t i = 0; i < out_grad_index.size(); ++i) {
      out_grad_index[i] = counter++;
    }
    for (size_t i = 0; i < in_data_index.size(); ++i) {
      in_data_index[i] = counter++;
    }
    for (size_t i = 0; i < out_data_index.size(); ++i) {
      out_data_index[i] = counter++;
    }
    std::vector<T> all_data;
    all_data.insert(all_data.end(), out_grad.begin(), out_grad.end());
    all_data.insert(all_data.end(), in_data.begin(), in_data.end());
    all_data.insert(all_data.end(), out_data.begin(), out_data.end());

    std::vector<int> ret_index = this->DeclareBackwardDependency(
        out_grad_index, in_data_index, out_data_index);

    std::vector<T> ret(ret_index.size());
    for (size_t i = 0; i < ret_index.size(); ++i) {
      ret[i] = all_data[ret_index[i]];
    }
    return ret;
  }
  /*!
   * \brief create OperatorProperty
   * \param type_name the type string of the OperatorProperty
   * \return a new constructed OperatorProperty
   */
  static OperatorProperty *Create(const char* type_name);

  virtual void Save(dmlc::JSONWriter *writer) const = 0;
  virtual void Load(dmlc::JSONReader *reader) = 0;
};

/*! \brief typedef the factory function of operator property */
typedef OperatorProperty *(*OperatorPropertyFactory)();
/*!
 * \brief Registry entry for OperatorProperty factory functions.
 */
struct OperatorPropertyReg
    : public dmlc::FunctionRegEntryBase<OperatorPropertyReg,
                                        OperatorPropertyFactory> {
  /*!
   * \brief Set key_var_num_args
   *  When this is set, the API caller is required to pass in a
   *  argument with key=key_num_args.c_str(), and value=num_args.
   *  num_args is number of positional argument when calling the function.
   *
   *  This is used to pass in length of positional arguments
   *  for operators that can take variable length of input.
   *  Most operators do not need to set this property.
   *
   * \param key the key name to be set
   */
  inline OperatorPropertyReg& set_key_var_num_args(const std::string &key) {  // NOLINT(*)
    this->key_var_num_args = key;
    return *this;
  }
  /*! \brief The key num_args name. */
  std::string key_var_num_args;
};

//--------------------------------------------------------------
// The following part are API Registration of Operators
//--------------------------------------------------------------
/*!
 * \brief Macro to register OperatorProperty
 *
 * \code
 * // example of registering a fully connected operator
 * REGISTER_OP_PROPERTY(FullyConnected, FullyConnectedOpProp)
 * .describe("Fully connected layer");
 *
 * \endcode
 */
#define MXNET_REGISTER_OP_PROPERTY(name, OperatorPropertyType)          \
  static ::mxnet::OperatorProperty* __create__ ## OperatorProperty ## name ## __() { \
    return new OperatorPropertyType;                                    \
  }                                                                     \
  DMLC_REGISTRY_REGISTER(::mxnet::OperatorPropertyReg, OperatorPropertyReg, name) \
  .set_body(__create__ ## OperatorProperty ## name ## __)

#endif  // DMLC_USE_CXX11
}  // namespace mxnet
#endif  // MXNET_OPERATOR_H_
