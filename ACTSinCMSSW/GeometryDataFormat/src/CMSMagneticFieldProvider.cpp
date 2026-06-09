#include "ACTSinCMSSW/GeometryDataFormat/interface/CMSMagneticFieldProvider.hpp"

CMSMagneticFieldProvider::CMSMagneticFieldProvider(const MagneticField& bField): bField_(bField) {};

Acts::MagneticFieldProvider::Cache CMSMagneticFieldProvider::makeCache(const Acts::MagneticFieldContext& mctx) const 
{
    return Cache{};
}

Acts::Result<Acts::Vector3> CMSMagneticFieldProvider::getField(const Acts::Vector3& position, Cache& cache) const {
    GlobalPoint point(position[0] / 10, position[1] / 10, position[2] / 10); // from mm to cm for CMSSW
    GlobalVector field = bField_.inTesla(point) * Acts::UnitConstants::T;

    return Acts::Result<Acts::Vector3>::success(Acts::Vector3(field.x(), field.y(), field.z()));
}